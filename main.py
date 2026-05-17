import sys
import pathlib
import warnings

# Suppress PyTorch FutureWarnings
warnings.filterwarnings("ignore")

base_path = pathlib.Path(__file__).parent
import forensics
from resnet50 import resnet50 as resnet_model
from rnn import rnn as rnn_model
from densenet121 import densenet as densenet_model

import torch

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    image_path = input("Please insert your image path: ").strip(' "\'')
    #image_path = pathlib.Path(r"C:\Users\faisa\Downloads\t7ccN51j9LinZJvRFxlQxN9W5foq_z2w.jpeg")
    results_path = base_path / "results"
    
    if not results_path.exists():
        results_path.mkdir()

    # Redirect output to file as well
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(results_path / "output.txt")

    print(f"==================================================")
    print(f"--- DEEPFAKE DETECTION SYSTEM: EXTENDED MODE ---")
    print(f"==================================================\n")
    print(f"TARGET IMAGE: {image_path}\n")
    
    # 1. ANALYZE METADATA
    print("[STEP 1] Extracting and Analyzing Metadata...")
    metadata = forensics.get_metadata(image_path)
    
    if metadata:
        anomalies = forensics.analyze_metadata_anomaly(metadata)
        if anomalies:
            print("\n!!! METADATA ANOMALIES DETECTED !!!")
            for anomaly in anomalies:
                print(f"[!] {anomaly}")
        else:
            print("\nStatus: Metadata consistency check passed.")
    else:
        print("\nStatus: Metadata extraction failed.")

    # 2. PERFORM ERROR LEVEL ANALYSIS (ELA)
    print("\n" + "="*50)
    print("[STEP 2] Performing Error Level Analysis (ELA)...")
    ela_image = forensics.perform_ela(image_path)
    
    if ela_image:

        if not (base_path / "results").exists():
            (base_path / "results").mkdir()

        output_name = base_path / "results" / "ela_result.png"
        ela_image.save(output_name)
        print(f"\nSUCCESS: ELA map generated and saved to '{output_name}'")
        print("\nHOW TO INTERPRET:")
        print("-----------------")
        print("- Uniform areas (mostly dark or consistent noise): Likely authentic.")
        print("- Bright/High-contrast spots: May indicate localized editing or deepfake artifacts.")
        print("- Edges naturally show more difference than flat areas.")
    else:
        print("\nERROR: Could not perform ELA.")
    
    # 3. PERFORM BLUR DETECTION
    print("\n" + "="*50)
    print("[STEP 3] Performing Blur Detection...")
    blur_score = forensics.blur_Detection(image_path)
    print(f"\nBlur Score: {blur_score}")
    print("\nHOW TO INTERPRET:")
    print("-----------------")
    print("- High blur score: Image is likely authentic (typically over 100 depending on the image size).")
    print("- Low blur score: Image may be a deepfake (typically under 100 depending on the image size).")
    print("\n" + "="*50)
    
    # 4. PERFORM AI CLASSIFICATION (RESNET & RNN)
    print("[STEP 4] Performing AI Classification...")
    
    print("\n--- ResNet Analysis ---")
    try:
        resnet_model_path = base_path / "resnet50" / "resnet50.pth"
        resnet_model.predict_image(image_path, str(resnet_model_path))
    except Exception as e:
        print(f"[!] Could not perform ResNet50 classification: {e}")

    print("\n--- RNN Analysis ---")
    try:
        rnn_model_path = base_path / "rnn" / "rnn.pth"
        rnn_model.predict_image(image_path, str(rnn_model_path))
    except Exception as e:
        print(f"[!] Could not perform RNN classification: {e}")

    print("\n--- DenseNet Analysis ---")
    try:
        densenet_model_path = base_path / "densenet121" / "densenet.pth"
        densenet_model.predict_image(image_path, str(densenet_model_path))
    except Exception as e:
        print(f"[!] Could not perform DenseNet classification: {e}")

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE.")