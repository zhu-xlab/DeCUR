### SAR-MS pretraining on SSL4EO-S12 dataset

Due to inode and I/O bottleneck, we download and convert the raw SSL4EO-S12 dataset to the lmdb format. As ablated in [SSL4EO-S12](https://arxiv.org/abs/2211.07044) paper, compressed 8-bit S2 data achieved similar performance as raw 16-bit but saved a lot of storage and time. Therefore, we normalize both SAR and MS data based on dataset mean and std and save them in uint8.

For SAR:

```bash
python ssl4eo_dataset.py \
--root /path/to/SSL4EO-S12 \
--save_path /path/to/ssl4eo-s12_0k_251k_sar.lmdb \
-- make_lmdb_file \
--num_workers 8 \
--normalize \
--mode s1 \
--dtype uint8 \
```

For MS:

```bash
python ssl4eo_dataset.py \
--root /path/to/SSL4EO-S12 \
--save_path /path/to/ssl4eo-s12_0k_251k_s2c_norm.lmdb \
-- make_lmdb_file \
--num_workers 8 \
--normalize \
--mode s2c \
--dtype uint8 \
```

Then, we build pytorch dataset class `LMDBDataset` in `ssl4eo_dataset_lmdb_mm_norm.py` to load data from the lmdb files.