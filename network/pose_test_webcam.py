from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    print("cuda : available")
    model.half().to(device)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, image = cam.read()    
    start = time.time()
    image = letterbox(image, 1280, stride=64, auto=True)[0]
    with torch.no_grad():
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        if torch.cuda.is_available():
            image = image.half().to(device)
        output, _ = model(image)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    if output.shape[0] > 0:
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    end = time.time()
    fps = 1 - (end-start)
    start = end
    cv2.putText(nimg, "FPS : %0.3f"%fps, (7,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,0), 3, cv2.LINE_AA)
    cv2.putText(nimg, "%d detected"%output.shape[0], (7,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,0), 3, cv2.LINE_AA)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    cv2.imshow('pose', nimg)
    if cv2.waitKey(1) == ord('q'):
        break
    
    del nimg, output, image
    torch.cuda.empty_cache()

cam.release()
cv2.destroyAllWindows()
