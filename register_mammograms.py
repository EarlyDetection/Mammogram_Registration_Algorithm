import numpy as np
import cv2
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
import os
import argparse
import matplotlib.pyplot as plt

def normalizeImage(img):
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img

# Mirror RMLO and RCC mammograms 
def left_orient_mammogram(img):
    left_nonzero = cv2.countNonZero(img[:, 0:int(img.shape[1]/2)])
    right_nonzero = cv2.countNonZero(img[:, int(img.shape[1]/2):])
    
    if(left_nonzero < right_nonzero):
        img = cv2.flip(img, 1)
    return img

# SIFT + Homography   
def registerImages(OGP, SCP):
    img1_8u = normalizeImage(SCP)
    img2_8u = normalizeImage(OGP)

    sift = cv2.SIFT_create(contrastThreshold=-1)

    kp1, des1 = sift.detectAndCompute(img1_8u, None)
    kp2, des2 = sift.detectAndCompute(img2_8u, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = good_matches[:250]

    if len(good_matches) < 4:
        print("Insufficient matches to compute homography.")
        return None

    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width = OGP.shape
    
    original_dtype = SCP.dtype
    img1REG = cv2.warpPerspective(SCP, h.astype(np.float32), (width, height), flags=cv2.INTER_LINEAR)

    return img1REG.astype(original_dtype)

def showImages(og_img, sc_img, registered_sc_img, filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Registration Results for: {filename}", fontsize=16)

    axes[0].imshow(og_img, cmap='gray')
    axes[0].set_title("Original Mammogram")
    axes[0].axis('off')

    axes[1].imshow(sc_img, cmap='gray')
    axes[1].set_title("Secondary Capture")
    axes[1].axis('off')

    axes[2].imshow(registered_sc_img, cmap='gray')
    axes[2].set_title("Registered Secondary Capture")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def process_and_register(secondary_capture_dir, original_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(secondary_capture_dir) if f.endswith('.dcm')]
    print(f"[INIT] Found {len(files)} DICOM files in {secondary_capture_dir}")

    for file in files:
        sc_path = os.path.join(secondary_capture_dir, file)
        og_path = os.path.join(original_dir, file)

        if not os.path.exists(og_path):
            print(f"[SKIP] Original file not found for {file}")
            continue

        print(f"Registering {file}...")
        
        try:
            SC = pydicom.dcmread(sc_path, force=True)
            OG = pydicom.dcmread(og_path, force=True)

            SCP = SC.pixel_array
            OGP = OG.pixel_array   
            
            SCP = left_orient_mammogram(SCP)
            OGP = left_orient_mammogram(OGP)

            registered_sc = registerImages(OGP, SCP)

            if registered_sc is not None:
                showImages(OGP, SCP, registered_sc, file)

                SC.Rows, SC.Columns = registered_sc.shape
                
                SC.PixelData = registered_sc.tobytes()
                
                SC.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                SC.is_little_endian = True
                SC.is_implicit_VR = False

                out_path = os.path.join(output_dir, file)
                SC.save_as(out_path)
                print(f"[SUCCESS] Saved registered DICOM to {out_path}\n")
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mammogram SIFT Registration")

    parser.add_argument(
        "--secondary_capture_dir", 
        type=str, 
        required=True, 
        help="Path to the folder containing Secondary Capture DICOM files."
    )
    parser.add_argument(
        "--original_dir", 
        type=str, 
        required=True, 
        help="Path to the folder containing Original DICOM files."
    )

    args = parser.parse_args()
    
    output_directory = "secondary_capture_registered"

    print(f"Starting processing...")
    print(f"Secondary Capture Directory: {args.secondary_capture_dir}")
    print(f"Original Directory: {args.original_dir}")
    print(f"Output Directory: {output_directory}")
    
    process_and_register(args.secondary_capture_dir, args.original_dir, output_directory)