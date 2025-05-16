To launch training on a Windows machine with a single GPU, instead of the bash script, run the following command directly in the terminal after activating the virtual environment:

```bash
New-Item -ItemType Directory -Path exp/knotty_captcha -Force; python train.py --epoch 60 --encoder vitl --bs 1 --lr 0.0000005 --save-path exp/knotty_captcha --dataset knotty_captcha --img-size 518 --min-depth 0.001 --max-depth 20 --pretrained-from ../checkpoints/depth_anything_v2_metric_hypersim_vitl.pth --port 20596 *>&1 | Tee-Object -FilePath exp/knotty_captcha/$(Get-Date -Format 'yyyyMMdd_HHmmss').log
```

Make any modifications to the above command as needed. Specifically, you should probably adjust the learning rate (make smaller for further finetuning), the number of epochs, the dataset, the encoder, and the batch size.

You can also choose to use the non-metric variants of the models (named like `depth_anything_v2_{encoder}.pth`) where `encoder` is one of `vits`, `vitb`, `vitl`, or `vitg` (currently unreleased).
