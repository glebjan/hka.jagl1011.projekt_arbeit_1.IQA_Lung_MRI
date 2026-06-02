from pathlib import Path

RESNET50 = Path("models/RadImageNet_pytorch/ResNet50.pt")

REPORT = Path("report") / "IXI661_report.csv"

INPUT  = Path("data/smore/coronal/IXI661-HH-2788-T1/IXI661-HH-2788-T1_smore4.nii.gz")
TARGET = Path("data/IXI_test_resampled/IXI661-HH-2788-T1.nii.gz")
