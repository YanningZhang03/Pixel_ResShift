#!/usr/bin/env python3
"""
ImageNet-1k Dataset Preparation Tool

Downloads, extracts, and organizes the ImageNet-1k dataset into
the standard directory structure required for training.

Reference: https://blog.csdn.net/qq_45588019/article/details/125642466
"""

import os
import subprocess
import shutil
import tarfile
from scipy import io
import sys


class ImageNetProcessor:
    def __init__(self, data_dir="/root/data/imagenet_data"):
        self.data_dir = data_dir
        self.train_tar = "ILSVRC2012_img_train.tar"
        self.val_tar = "ILSVRC2012_img_val.tar"
        self.devkit_tar = "ILSVRC2012_devkit_t12.tar.gz"
        os.makedirs(data_dir, exist_ok=True)

    def _download_with_aria2(self, url: str, out_path: str, connections: int = 4, splits: int = 4):
        """Download a file using aria2c with multi-connection and resume support."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        cmd = [
            "aria2c",
            "-c",                        # resume download
            "-x", str(connections),      # max connections per server
            "-s", str(splits),           # number of splits
            "-k", "1M",                  # min split size
            "--allow-overwrite=true",
            "--check-certificate=false",
            "-o", os.path.basename(out_path),
            "-d", os.path.dirname(out_path),
            url,
        ]
        subprocess.run(cmd, check=True)

    def download_dataset(self):
        """Download ImageNet-1k dataset files."""
        print("Downloading ImageNet-1k dataset...")

        train_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
        print(f"Downloading training set: {self.train_tar}")
        self._download_with_aria2(train_url, os.path.join(self.data_dir, self.train_tar))

        val_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
        print(f"Downloading validation set: {self.val_tar}")
        self._download_with_aria2(val_url, os.path.join(self.data_dir, self.val_tar))

        devkit_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
        print(f"Downloading label mapping: {self.devkit_tar}")
        self._download_with_aria2(devkit_url, os.path.join(self.data_dir, self.devkit_tar))

        print("All files downloaded successfully!")
    
    def extract_train_set(self):
        """Extract the training set and organize into per-class directories."""
        print("Extracting training set...")
        
        train_dir = os.path.join(self.data_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        
        train_tar_path = os.path.join(self.data_dir, self.train_tar)
        print(f"Extracting main training archive to: {train_dir}")
        subprocess.run([
            "tar", "-xvf", train_tar_path, "-C", train_dir
        ], check=True)
        
        original_cwd = os.getcwd()
        os.chdir(train_dir)
        
        try:
            tar_files = [f for f in os.listdir('.') if f.endswith('.tar')]
            print(f"Found {len(tar_files)} per-class tar files")
            
            for tar_file in tar_files:
                class_name = tar_file.replace('.tar', '')
                class_dir = os.path.join(train_dir, class_name)
                
                os.makedirs(class_dir, exist_ok=True)
                
                print(f"Extracting class: {class_name}")
                subprocess.run([
                    "tar", "-xvf", tar_file, "-C", class_dir
                ], check=True)
                
                os.remove(tar_file)
                
        finally:
            os.chdir(original_cwd)
        
        self._verify_train_extraction(train_dir)
    
    def _verify_train_extraction(self, train_dir):
        """Verify the training set extraction results."""
        print("Verifying training set extraction...")
        
        result = subprocess.run([
            "bash", "-c", f"cd {train_dir} && ls -lR | grep '^d' | wc -l"
        ], capture_output=True, text=True)
        folder_count = int(result.stdout.strip())
        print(f"Class folders: {folder_count} (expected: 1000)")
        
        result = subprocess.run([
            "bash", "-c", f"cd {train_dir} && ls -lR | grep '^-' | wc -l"
        ], capture_output=True, text=True)
        file_count = int(result.stdout.strip())
        print(f"Image files: {file_count} (expected: 1,281,167)")
    
    def extract_validation_set(self):
        """Extract the validation set and classify images into per-class directories."""
        print("Extracting validation set...")
        
        val_dir = os.path.join(self.data_dir, "val")
        os.makedirs(val_dir, exist_ok=True)
        
        val_tar_path = os.path.join(self.data_dir, self.val_tar)
        print(f"Extracting validation archive to: {val_dir}")
        subprocess.run([
            "tar", "xvf", val_tar_path, "-C", val_dir
        ], check=True)
        
        devkit_path = os.path.join(self.data_dir, self.devkit_tar)
        print("Extracting label mapping (devkit)...")
        subprocess.run([
            "tar", "-xzf", devkit_path, "-C", self.data_dir
        ], check=True)
        
        self._classify_validation_set(val_dir)
    
    def _classify_validation_set(self, val_dir):
        """Sort validation images into per-class directories using ground truth labels."""
        print("Classifying validation images...")
        
        devkit_dir = os.path.join(self.data_dir, "ILSVRC2012_devkit_t12")
        
        synset = io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))
        
        ground_truth_path = os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt')
        with open(ground_truth_path, 'r') as f:
            lines = f.readlines()
        labels = [int(line.strip()) for line in lines]
        
        for filename in os.listdir(val_dir):
            if not filename.endswith('.JPEG'):
                continue
                
            val_id = int(filename.split('.')[0].split('_')[-1])
            
            ILSVRC_ID = labels[val_id - 1]  # labels are 1-indexed
            WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]  # class name
            
            print(f"Processing: {filename}, val_id:{val_id}, ILSVRC_ID:{ILSVRC_ID}, class:{WIND}")
            
            output_dir = os.path.join(val_dir, WIND)
            os.makedirs(output_dir, exist_ok=True)
            
            src_path = os.path.join(val_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            shutil.move(src_path, dst_path)
        
        print("Validation set classification complete!")
        
        self._verify_val_classification(val_dir)
    
    def _verify_val_classification(self, val_dir):
        """Verify the validation set classification results."""
        print("Verifying validation set classification...")
        
        folders = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
        print(f"Validation class folders: {len(folders)} (expected: 1000)")
        
        total_images = 0
        for folder in folders:
            folder_path = os.path.join(val_dir, folder)
            images = [f for f in os.listdir(folder_path) if f.endswith('.JPEG')]
            total_images += len(images)
        
        print(f"Validation images total: {total_images} (expected: 50,000)")


def main():
    print("=" * 60)
    print("ImageNet-1k Dataset Preparation Tool")
    print("=" * 60)
    
    processor = ImageNetProcessor()
    
    while True:
        print("\nSelect an operation:")
        print("1. Download dataset")
        print("2. Extract training set")
        print("3. Extract and classify validation set")
        print("4. Full pipeline (download + extract + classify)")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            try:
                processor.download_dataset()
            except Exception as e:
                print(f"Download failed: {e}")
                
        elif choice == "2":
            try:
                processor.extract_train_set()
            except Exception as e:
                print(f"Training set extraction failed: {e}")
                
        elif choice == "3":
            try:
                processor.extract_validation_set()
            except Exception as e:
                print(f"Validation set extraction failed: {e}")
                
        elif choice == "4":
            try:
                print("Starting full pipeline...")
                processor.download_dataset()
                processor.extract_train_set()
                processor.extract_validation_set()
                print("Full pipeline complete!")
            except Exception as e:
                print(f"Full pipeline failed: {e}")
                
        elif choice == "5":
            print("Exiting.")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()
