# Setup file paths

<details><summary>COCO</summary>

Update the following paths in `yaml/coco.yml`:

- `downloads_dir`: directory where COCO data would be downloaded
- `proc_dir`: directory where processed COCO data would be stored
- `exp_dir`: directory where COCO experiment runs would be saved
- `image_dir`: directory where COCO images would be extracted
- `local_proc_dir`: a local copy of `proc_dir` if frequent reads from `proc_dir` is a problem. This is useful, for example, if `proc_dir` is NFS shared across multiple machines and `local_proc_dir` is local data storage for the machine you want to run experiments on. We provide scripts for copying files from `proc_dir` to `local_proc_dir`.

In my setup `downloads_dir`, `proc_dir`, and `exp_dir` are directories on a shared NFS storage while `image_dir` and `local_proc_dir` point to local storage. 
</details>


# Download and extract data

<details><summary>COCO</summary>

Before running the following, please make sure the paths are correctly setup in `yaml/coco.yml`.
```python
# download COCO images and annotations to downloads_dir
python -m data.coco.download
# extract annotations to coco_proc
python -m data.coco.extract_annos
# extract images to image_dir
python -m data.coco.extract_images
```
</details>


# Get object detections

We provide detections for COCO and Flickr30K images computed using a FasterRCNN model trained on VisualGenome object and attribute annotations originally used in the [Bottom-Up and Top-Down Attention](https://arxiv.org/abs/1707.07998) work and then reused in a recent weakly supervised phrase grounding work [Align2Ground](https://arxiv.org/abs/1903.11649) that we compare to. 

We use a [lightly modified fork](https://github.com/BigRedT/bottom-up-features) of the pytorch implementation available [here](https://github.com/violetteshev/bottom-up-features) to extract bounding boxes, scores, and features from a set of images and save them in hdf5 format. 

Download and extract detections to a desired location:
- [COCO](https://drive.google.com/file/d/1I70cDM2MEe56tZVq8PELffm3S13uYjHV/view?usp=sharing) [20 GB]
- [Flickr30K](https://drive.google.com/file/d/1CxTY38nKPFe9wikEdpeU-XXMwbFs77GV/view?usp=sharing) [5 GB]


# Construct context-preserving negative captions

<details><summary>COCO</summary>

**Step 1:** Identity noun tokens to be substituted
```
bash exp/gen_noun_negatives/scripts/identify_tokens.sh train
bash exp/gen_noun_negatives/scripts/identify_tokens.sh val
```
This creates the following files in `<proc_dir>/annotations`:
- `noun_tokens_<subset>.json`: identified noun tokens in captions
- `noun_vocab_<subset>.json`: noun vocabulary

**Step 2:** Sample substitute words 
```
bash exp/gen_noun_negatives/scripts/sample_neg_bert.sh train
bash exp/gen_noun_negatives/scripts/sample_neg_bert.sh val
```
This creates the following files in `<proc_dir>`:
- `bert_noun_negatives_<subset>.json`: contains negative captions constructed by substituting a word in the positive caption
- `vis_bert_noun_negatives_<subset>.html`: an webpage visualizing words tokens in the positive caption, the token replaced, top 30 negatives sampled from q(s|s',c) (`True Pred`), top 30 negatives sampled from p(s'|c) (`Lang Pred`), reranked Lang Pred negatives (`Rerank Pred`). The last 5 words in Rerank Pred are discarded and remaining 25 are used as negatives. Here's an example:  
![Screenshot of the webpage displaying sampled negatives](imgs/sampled_negatives.png)

**Step 3:** Cache contextualized representations of the substituted words
```
bash exp/gen_noun_negatives/scripts/cache_neg_fetures.sh train
bash exp/gen_noun_negatives/scripts/cache_neg_fetures.sh val
```
This creates the following files in `<proc_dir>`:

</details>

<details><summary>Flickr30K</summary>

</details>


# Learn to ground

Once we have the following, we are ready to train our grounding model:
- Detections on train and val sets for the dataset you want to train on (COCO or Flickr30K)
- Negatives with cached features for the train and val set for the same dataset

**Step 1:** Identify noun and adjective tokens to maximize mutual information with the image regions.
```bash
# For COCO
bash exp/ground/scripts/identify_noun_adj_tokens.sh train
bash exp/ground/scripts/identify_noun_adj_tokens.sh val

# For Flickr
bash exp/ground/scripts/identify_noun_adj_tokens_flickr.sh train
bash exp/ground/scripts/identify_noun_adj_tokens_flickr.sh val
```
This creates `<proc_dir>/annotations/noun_adj_tokens_<subset>.json`

**Step 2:** Copy over detections and cached features to `<local_proc_dir>`. This may reduce training time if, for instance, `<proc_dir>` is a slow shared NFS and `<local_proc_dir>` is a faster local drive. Otherwise you may skip this step and set `<local_proc_dir>` to the same path as `<proc_dir>`.

To copy, modify path variables `NFS_DATA` and `LOCAL_DATA` in `setup_coco.sh` or `setup_flickr.sh` and execute 
```bash
# For COCO
bash setup_coco.sh

# For Flickr
bash setup_flickr.sh
```

**Step 3:** Start training by executing
```bash
# For COCO
bash exp/ground/scripts/train.sh

# For Flickr
bash exp/ground/scripts/train_flickr.sh
```

# Evaluate on Flickr
To evaluate on Flickr, follow the instructions above to setup Flickr file paths, download/extract the dataset, and download object detections. If needed also run `setup_flickr.sh` to copy files from NFS to local disk after modifying `NFS_DATA` and `LOCAL_DATA` paths in the script.

**Model Selection:** As noted in our paper, we use ground truth annotations in the Flickr validation set for model selection. To perform model selection run
```bash
# For COCO
bash exp/ground/scripts/eval_flickr_phrase_loc_model_selection.sh model_trained_on_coco

# For Flickr
bash exp/ground/scripts/eval_flickr_phrase_loc_model_selection.sh model_trained_on_flickr
```

**Model evaluation:** To evaluate the selected model, run 
```bash
# For COCO
bash exp/ground/scripts/eval_flickr_phrase_loc.sh model_trained_on_coco

# For Flickr
bash exp/ground/scripts/eval_flickr_phrase_loc.sh model_trained_on_flickr
```



