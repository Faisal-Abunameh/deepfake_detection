import metadata_detection
import forensics
import pathlib
#hi im just testin
if __name__ == "__main__":

    image_path = input("Please insert your image path: ")
    #image_path = pathlib.Path(r"C:\Users\faisa\Downloads\t7ccN51j9LinZJvRFxlQxN9W5foq_z2w.jpeg")
    base_path = pathlib.Path(__file__).parent    
    print(f"==================================================")
    print(f"--- DEEPFAKE DETECTION SYSTEM: EXTENDED MODE ---")
    print(f"==================================================\n")
    print(f"TARGET IMAGE: {image_path}\n")
    
    # 1. ANALYZE METADATA
    print("[STEP 1] Extracting and Analyzing Metadata...")
    metadata = metadata_detection.get_metadata(image_path)
    
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
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE.")