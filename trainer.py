import os, sys
from decimal import Decimal
import utility
import torch
from utils.postprocessing_functions import SimplePostProcess
from utils.metrics import PSNR
import time
from torch.cuda.amp import autocast as autocast, GradScaler

train_log_dir = './experiment/train_log/'

exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
tfboard_name = exp_name + "_"
exp_train_log_dir = os.path.join(train_log_dir, exp_name)

LOG_DIR = os.path.join(exp_train_log_dir, 'logs')

# save img path
IMG_SAVE_DIR = os.path.join(exp_train_log_dir, 'img_log')
# Where to load model
LOAD_MODEL_DIR = os.path.join(exp_train_log_dir, 'models')
# Where to save new model
SAVE_MODEL_DIR = os.path.join(exp_train_log_dir, 'real_models')

SAVE_STATE_DIR = os.path.join(exp_train_log_dir, 'training_states')

# Where to save visualization images (for report)
RESULTS_DIR = os.path.join(exp_train_log_dir, 'report')

utility.mkdir(SAVE_STATE_DIR)
utility.mkdir(SAVE_MODEL_DIR)
utility.mkdir(IMG_SAVE_DIR)
utility.mkdir(LOG_DIR)


class Trainer():
    def __init__(self, args, train_loader, train_sampler, valid_loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale[0]

        self.ckp = ckp
        self.loader_train = train_loader
        self.loader_valid = valid_loader
        self.train_sampler = train_sampler
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        self.neg_num = args.neg_num

        ###################################
        if args.pre_train == "":
            self.fix_unflagged = True
        else:
            self.fix_unflagged = False
        self.fix_epoch = 5
        self.fix_keys = ["spynet", "dcnpack"]

        ###################################

        self.psnr_fn = PSNR(boundary_ignore=40)
        # Postprocessing function to obtain sRGB images
        self.postprocess_fn = SimplePostProcess(return_np=True)

        if self.args.fp16:
            self.scaler = GradScaler()

        self.best_psnr = 0.
        self.best_epoch = 1

        self.error_last = 1e8
        self.glob_iter = 0

        self.log_dir = LOG_DIR + "/" + args.save
        self.img_save_dir = IMG_SAVE_DIR + "/" + args.save
        # Where to load model
        self.load_model_dir = LOAD_MODEL_DIR + "/" + args.save
        # Where to save new model
        self.save_model_dir = SAVE_MODEL_DIR + "/" + args.save
        self.save_state_dir = SAVE_STATE_DIR + "/" + args.save

        # Where to save visualization images (for report)
        self.results_dir = RESULTS_DIR + "/" + args.save

        if self.args.load != '':
            self.optimizer.load(self.save_state_dir, epoch=int(self.args.load))

        utility.mkdir(self.save_state_dir)
        utility.mkdir(self.save_model_dir)
        utility.mkdir(self.img_save_dir)
        utility.mkdir(self.log_dir)
        utility.mkdir('frames')

        if self.args.local_rank <= 0:
            number_parameters = sum(map(lambda x: x.numel(), self.model.parameters()))
            print("number of parameters: ", number_parameters)

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        if self.train_sampler:
            self.train_sampler.set_epoch(epoch)
        if epoch % 1 == 0:
            self.ckp.write_log(
                '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
            )
        self.loss.start_log()

        # train alignment module after 5 epochs.
        if self.args.pre_train == "":
            if self.fix_unflagged and epoch <= self.fix_epoch:
                if self.args.local_rank <= 0:
                    print(f'Fix keys: {self.fix_keys} for the first {self.fix_epoch} epochs.')
                self.fix_unflagged = False
                for name, param in self.model.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        param.requires_grad_(False)
            elif epoch > self.fix_epoch:
                if self.args.local_rank <= 0:
                    print(f'Train all the parameters from {self.fix_epoch+1} epochs.')
                self.model.requires_grad_(True)

        self.model.train()
        if self.args.local_rank == 0:
            timer_data, timer_model, timer_epoch = utility.timer(), utility.timer(), utility.timer()
            timer_epoch.tic()
        for batch, batch_value in enumerate(self.loader_train):
            burst, gt, flow_vectors, meta_info = batch_value
            burst, gt, flow_vectors = self.prepare(burst, gt, flow_vectors)
            if self.args.local_rank == 0:
                timer_data.hold()
                timer_model.tic()

            if self.args.fp16:
                with autocast():
                    sr = self.model(burst, 0)
                    if self.neg_num > 0:
                        neg_sample = burst[torch.randperm(self.neg_num), :, :, :, :]
                        loss = self.loss(sr, gt, neg_sample)
                    else:
                        loss = self.loss(sr, gt, 0)
            else:
                sr = self.model(burst, 0)
                if self.neg_num > 0:
                    neg_sample = burst[torch.randperm(self.neg_num), 0, :, :, :]
                    neg_sample = torch.nn.Upsample(scale_factor=8, mode='bilinear')(neg_sample)
                    cha_align = torch.nn.Conv2d(neg_sample.shape[-3], 3, 3, 1, 1).to(
                        torch.device('cpu' if self.args.cpu else 'cuda:{}'.format(self.args.local_rank)))
                    neg_sample = cha_align(neg_sample)

                    loss = self.loss(gt, sr.to(torch.float32), neg_sample)
                else:
                    loss = self.loss(sr, gt, 0)

            if self.args.n_GPUs > 1:
                torch.distributed.barrier()
                reduced_loss = utility.reduce_mean(loss, self.args.n_GPUs)
            else:
                reduced_loss = loss

            self.optimizer.zero_grad()

            if self.args.fp16:
                self.scaler.scale(loss).backward()

                if torch.isinf(sr).sum() + torch.isnan(sr).sum() <= 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    print(f'Nan num: {torch.isnan(sr).sum()}, inf num: {torch.isinf(sr).sum()}')
                    os._exit(0)
                    sys.exit(0)
            else:
                loss.backward()
                if torch.isinf(sr).sum() + torch.isnan(sr).sum() <= 0:
                    self.optimizer.step()
                else:
                    print(f'Nan num: {torch.isnan(sr).sum()}, inf num: {torch.isinf(sr).sum()}')
                    os._exit(0)
                    sys.exit(0)

            if self.args.local_rank == 0:
                timer_model.hold()
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log('[{}/{}]\t[{:.4f}]\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        reduced_loss.item(),
                        timer_model.release(),
                        timer_data.release()))

                self.glob_iter += 1
                timer_data.tic()

            if self.args.local_rank <= 0 and (batch + 1) % 2000 == 0:
                if not self.args.test_only:
                    filename = exp_name + '_latest' + '.pth'
                    self.save_model(filename)

        if self.args.local_rank <= 0:
            timer_epoch.hold()
            print('Epoch {} cost time: {:.1f}s, lr: {:5f}'.format(epoch, timer_epoch.release(), lr))
            if (epoch) % 1 == 0 and not self.args.test_only:
                filename = exp_name + '_epoch_' + str(epoch) + '.pth'
                self.save_model(filename)

            if not self.args.test_only:
                filename = exp_name + '_latest' + '.pth'
                self.save_model(filename)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.test()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self, print_time=False):

        def ttaup(burst):
            return [burst]

        def ttadown(bursts):
            burst0 = bursts[0]
            out = burst0
            return out

        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch() + 1
        self.model.eval()
        if self.args.local_rank == 0:
            timer_test = utility.timer()
        if epoch == 1 or epoch % 1 == 0:
            self.model.eval()
            total_psnr = 0
            total_ssim = 0
            total_lpips = 0
            count = 0
            if self.args.local_rank <= 0:
                print("Testing...")
            for i, batch_value in enumerate(self.loader_valid):
                burst_, gt, meta_info = batch_value
                burst_, gt = self.prepare(burst_, gt)

                bursts = ttaup(burst_)

                if print_time and self.args.local_rank <= 0:
                    tic = time.time()
                with torch.no_grad():
                    srs = []
                    for burst in bursts:
                        if self.args.fp16:
                            with autocast():
                                sr = self.model(burst, 0).float()
                        else:
                            sr = self.model(burst, 0).float()
                        srs.append(sr)
                    sr = ttadown(srs)

                if print_time and self.args.local_rank <= 0:
                    toc = time.time()
                    print(f'model pass time: {toc - tic:.4f}')

                psnr_score, ssim_score, lpips_score = self.psnr_fn(sr, gt)

                if self.args.n_GPUs > 1:
                    torch.distributed.barrier()
                    psnr_score = utility.reduce_mean(psnr_score, self.args.n_GPUs)
                    ssim_score = utility.reduce_mean(ssim_score, self.args.n_GPUs)
                    lpips_score = utility.reduce_mean(lpips_score, self.args.n_GPUs)

                total_psnr += psnr_score
                total_ssim += ssim_score
                total_lpips += lpips_score
                count += 1

            total_psnr = total_psnr / count
            total_ssim = total_ssim / count
            total_lpips = total_lpips / count
            if self.args.local_rank == 0:
                if total_psnr > self.best_psnr:
                    self.best_psnr = total_psnr
                    self.best_epoch = epoch
                    filename = exp_name + '_best_epoch.pth'
                    self.save_model(filename)
                print("[Epoch: {}][PSNR: {:.4f}][SSIM: {:.4f}][LPIPS: {:.4f}][Best PSNR: {:.4f}][Best Epoch: {}]"
                      .format(epoch, total_psnr, total_ssim, total_lpips, self.best_psnr, self.best_epoch))

                print('Forward: {:.2f}s\n'.format(timer_test.toc()))

        torch.cuda.synchronize()
        torch.set_grad_enabled(True)
        torch.cuda.empty_cache()

    def save_model(self, filename):
        print('save model...')
        net_save_path = os.path.join(self.save_model_dir, filename)

        model = self.model.model
        if self.args.n_GPUs > 1:
            model = model.module

        torch.save(model.state_dict(), net_save_path)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda:{}'.format(self.args.local_rank))

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs