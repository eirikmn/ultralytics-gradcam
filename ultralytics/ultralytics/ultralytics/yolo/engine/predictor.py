# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolov8n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                              yolov8n.torchscript        # TorchScript
                              yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolov8n_openvino_model     # OpenVINO
                              yolov8n.engine             # TensorRT
                              yolov8n.mlmodel            # CoreML (macOS-only)
                              yolov8n_saved_model        # TensorFlow SavedModel
                              yolov8n.pb                 # TensorFlow GraphDef
                              yolov8n.tflite             # TensorFlow Lite
                              yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8n_paddle_model       # PaddlePaddle
"""
import platform
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.data.augment import LetterBox, classify_transforms
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, MACOS, SETTINGS, WINDOWS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.loss import BboxLoss, v8DetectionLoss2
from ultralytics.yolo.utils.lossfunc import v8detectionlosscomputer
from ultralytics.yolo.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh

### added by me ###
from ultralytics.yolo.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8PoseLoss, v8SegmentationLoss

###################


STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""



class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        print("\n\n basepredictor init \n\n")
        self.args = get_cfg(cfg, overrides)
        self.save_dir = self.get_save_dir()
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def get_save_dir(self):
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        return increment_path(Path(project) / name, exist_ok=self.args.exist_ok)

    def cam_show_img(self, img, feature_map, grads, out_name):
        #print("\n\ncam_show_img \n\n")
        #print(img.shape)
        H = img.shape[2]
        W = img.shape[3]
        #_, __, H, W = img.shape
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		
        grads = grads.reshape([grads.shape[0],-1])					
        weights = np.mean(grads, axis=1)							
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]							
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.resize(cam, (W, H))

        out_np_name = out_name.rsplit(".", 1)[0]
        np.save(out_np_name+".npy", cam)

        #print(cam.shape)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        #### edits by me ####
        #print("\n\n cam img heatmap type \n\n")
        #print(heatmap.shape)
        #print(type(heatmap))
        #heatmapp = torch.from_numpy(heatmap)
        #heatmapp = heatmapp.permute((2,0,1))
        #heatmap = heatmapp.permute((1,2,0))
        
        imgg = img[0].detach().cpu().numpy()
        imgg = np.transpose(imgg,(1,2,0))
        #imgg = img[0]
        #print("imgg")
        #print(type(imgg))
        #print(imgg.shape)
        #print(img[0].shape)
        #print(img[0,:,:,:].shape)
        cam_img = 0.3 * heatmap + 0.7 * imgg

        #print(out_name)
        #print(cam_img.shape)
        
        cv2.imwrite(out_name, cam_img)

        #export heatmap as a text file
        #print(cam_img.shape)
        
        

        #cv2.imwrite(out_name+"a", cam_img) #EMN-debugging: is this needed?
        
    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def pre_transform(self, im):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes and self.model.pt
        return [LetterBox(self.imgsz, auto=auto, stride=self.model.stride)(image=x) for x in im]

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)

        return log_string

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        return preds
    
    def init_criterion(self):
        print("\n\nv8DetectionLoss\n")
        print(type(self))
        print(type(self.model))
        print(type(self.model.parameters()))
        print("\n\n")
        return v8DetectionLoss(self.model.model)

    def compute_loss(self, preds, targets):
        print("\n\nv8DetectionLoss\n")
        print(self.args.__dict__.keys())
        print("\n\n")
        print(self.__dict__.keys())
        print("\n\n")
        print(self.model.__dict__.keys())
        print("\n\n")
        print(self.model.model.__dict__.keys())
        print("\n\n")
        #return v8DetectionLoss2(self)
        return v8detectionlosscomputer(self, preds, targets, self.device)


    def __call__(self, source=None, model=None, stream=False):
        """Performs inference on an image or stream."""
        print("\n\n basepredictor __call__ \n")
        print(type(self.model))
        print("\n\n")
        self.stream = stream
        if stream:
            return self.stream_inference(source, model)
        else:
            return list(self.stream_inference(source, model))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode."""
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs



    #function that reads the YOLO style labels from a txt file into a torch tensor
    def labelreader(self, labpath=None):
        labels = []
        with open(labpath, 'r') as f:
            for line in f:
                #append each line to the labels as floats
                labels.append("0 " + line.strip())

        for i in range(len(labels)):
            labels[i] = labels[i].split(' ')
            for j in range(len(labels[i])):
                labels[i][j] = float(labels[i][j])

        #cast each element to float and store to tensor
        labels = torch.tensor(labels)
        return labels


    #@smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        """Streams real-time inference on camera feed and saves results to file."""
        print("\n\n basepredictor stream_inference \n\n")
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)

        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        ##############################################################################################################
        
        if self.args.visualize:
            print("\n\n req grad True \n\n")
            print(type(self.model))
            # require grad
            for k, v in self.model.named_parameters(): #EMN: model does not have named_parameters apparently...
                v.requires_grad = True  # train all layers
            #compute_loss = ComputeLoss(model)
            print("\n\n req grad True1 \n\n")

            #compute_loss = self.init_criterion()
            #compute_loss = self.loss()
            #compute_loss = v8DetectionLoss2(self.model.model)
            print("\n\n req grad True2 \n\n")
        
        ############################################################################################################################


        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        # Checks
        if self.source_type.tensor and (self.args.save or self.args.save_txt or self.args.show):
            LOGGER.warning("WARNING ⚠️ 'save', 'save_txt' and 'show' arguments not enabled for torch.Tensor inference.")

        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        self.run_callbacks('on_predict_start')
        counter = 0
        for batch in self.dataset:
            counter += 1
            print("\n\n batch#")
            print(counter)
            print("\n\n")
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path[0]).stem,
                                       mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)
                print("\n\n batch reqgrad\n")
                #print(im0s.requires_grad)
                print(im.requires_grad)
                im.requires_grad = True
                #print("height")
                #print(im.requires_grad)
                #print("height")
            # Inference
            with profilers[1]:

                ##############################################################
                
                if self.args.visualize:
                    
                    # grad-cam
                    #for k, v in self.model.named_parameters(): #EMN: model does not have named_parameters apparently...
                        #print("v\n")
                        #print(v.shape)
                        #print(v.requires_grad)  # train all layers

                    preds = self.model(im, augment=self.args.augment, visualize=visualize)
                    
                    #print(preds[1].shape)
                    #print(preds)
                    #pred = model(img, augment=augment, visualize=visualize)
                    
                    

                    self.model.zero_grad()
                    
                    #print("\n\n backpropprop \n\n")
                    #print(preds[0].shape)
                    #print(len(preds[1]))
                    #print(preds[1][0].shape)
                    #print(preds[1][1].shape)
                    #print(preds[1][2].shape)

                    #targets = torch.zeros(2, 6)
                    #print(self.args)
                    #print(type(self.args))
                    #print(self.args.__dict__.keys())
                    labpath = "/Users/emy016/Dropbox/Postdoc2/Kurs/NORA summer school 2023/NORAprosjekt/sampleimg-3.txt"
                    #print(labpath)
                    targets = self.labelreader(labpath=labpath)
                    #print(targets.shape)
                    #print(len(targets))
                    #print(type(targets))
                    #print(targets[0])
                    #print("\n\n backpropprop \n\n")
                    
                    #targets = torch.zeros(2, 6) ###EMN: Finn ut av denne!!! 
                    
                    #create a target tensor with batch_idx, class, and bboxes filled with zeros
                    print("\n\n targettensor \n\n")
                    



                    #targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
                    print("\n\n backprop \n\n")
                    #loss, loss_items = compute_loss(preds, targets.to(self.args.device))
                    #loss, loss_items = loss(self.model, batch)
                    loss, loss_items = self.compute_loss(preds, targets)
                    print("\n computed loss \n")
                    loss.requires_grad_(True)
                    print("\n assigned req grad on loss \n")
                    loss.backward()
                    print("\n backward pass \n")


                    _grads = self.model.model.grads_list
                    _grads.reverse()
                    _features = self.model.model.features_list

                    # for g, f in zip(_grads, _features):
                    #     print('grad', type(g), g.shape)
                    #     print('feature', type(f), f.shape)
                    print("\n\n odd for loop \n\n")
                    
                    print("\n\n odd for loop \n\n")
                    #for i in [17, 18, 19, 20, 21, 22, 23, 24, 25]:
                    for i in range(24):
                        
                        out_name = str(self.save_dir / f"{i}.jpg")
                        #print("\n\n input arguments_ cam_show_img \n\n")
                        #print(im.shape)
                        #print(_features[i].cpu().detach().numpy()[0].shape)
                        #print(_grads[i].cpu().detach().numpy()[0].shape)
                        #print(out_name)
                        self.cam_show_img(im, _features[i].cpu().detach().numpy()[0], _grads[i].cpu().detach().numpy()[0], out_name)
                        #def cam_show_img(img, feature_map, grads, out_name)
                    preds = preds[0]
                else:
                    #pred = model(img, augment=augment, visualize=visualize)[0]
                    preds = self.model(im, augment=self.args.augment, visualize=visualize)
                    

                #preds = self.model(im, augment=self.args.augment, visualize=visualize) #EMN-debugging: is this needed?

            # Postprocess
            with profilers[2]:
                self.results = self.postprocess(preds, im, im0s)
            self.run_callbacks('on_predict_postprocess_end')

            print("\n\n visualize, save, write \n\n")
            # Visualize, save, write results
            n = len(im0s)
            for i in range(n):
                self.seen += 1
                self.results[i].speed = {
                    'preprocess': profilers[0].dt * 1E3 / n,
                    'inference': profilers[1].dt * 1E3 / n,
                    'postprocess': profilers[2].dt * 1E3 / n}
                p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                p = Path(p)

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))
                if self.args.save or self.args.save_txt:
                    self.results[i].save_dir = self.save_dir.__str__()
                if self.args.show and self.plotted_img is not None:
                    self.show(p)
                if self.args.save and self.plotted_img is not None:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))

            self.run_callbacks('on_predict_batch_end')
            yield from self.results

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *im.shape[2:])}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        print("\n\n basepredictor setup model \n")
        print(type(model))
        print("\n basepredictor setup model \n\n")
        device = select_device(self.args.device, verbose=verbose)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)
        self.device = device
        self.model.eval()

    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        im0 = self.plotted_img
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        # Save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix = '.mp4' if MACOS else '.avi' if WINDOWS else '.avi'
                fourcc = 'avc1' if MACOS else 'WMV2' if WINDOWS else 'MJPG'
                save_path = str(Path(save_path).with_suffix(suffix))
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """
        Add callback
        """
        self.callbacks[event].append(func)
