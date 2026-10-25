# Week 6: Land-cover classification with ResNet18

**Author:** Atteq ur Rehman | Applied AI, Lab 6 (individual)

**Who benefits:** analysts, planners and environmental agencies that use satellite-derived land-cover maps.
**Why it matters:** automatic classification of Sentinel-2 patches makes land-use monitoring faster and cheaper than manual mapping; a wrong class can mislead planning decisions.

## Datasheet

**Dataset:** EuroSAT (Helber et al., 2019), https://github.com/phelber/EuroSAT, Kaggle copy `apollo2506/eurosat-dataset`. 27,000 RGB Sentinel-2 images of 64 x 64 pixels in 10 classes; labels by the dataset authors; MIT licence (verify on the dataset page).

**Data audit:**
- Imbalance ratio 1.5 (2,000 to 3,000 images per class; Pasture is the smallest).
- 0 exact duplicates (MD5 over all 27,000 files).
- 0 of 3,750 test images flagged as near-duplicates at the 0.10 fingerprint threshold. However, the closest test/train pairs show shifted views of the same scene (same river bend, same road junction), so overlapping tiles may exist across splits.
- Split by group (file name; every image is its own group) into 18,750 train / 4,500 validation / 3,750 test images, stratified by class, with **0 shared groups**.

**Imbalance handling:** weighted cross-entropy, w_c = N / (K x n_c). Pasture 1.35, Highway / Industrial / PermanentCrop / River about 1.08, the five largest classes 0.90.

## Model card

**Model:** ResNet18 (ImageNet weights), 128 x 128 input. Phase 1: 2 epochs, head only (validation macro-F1 0.854). Phase 2: 3 epochs, last block unfrozen (validation macro-F1 0.962). Best epoch chosen by validation macro-F1.

**Results (test set, used once):** macro-F1 **0.960**, accuracy 0.961. Weakest class Pasture (recall 0.906, mostly confused with AnnualCrop); best classes Forest and Residential (recall 0.990). Binary threshold metrics (sensitivity, specificity, PPV) are not applicable because the task has 10 classes.

**Grad-CAM:** the model looks at image content (water, canopy, vegetation texture, linear features such as roads), not at borders or corners. No shortcut was confirmed, but reliance on colour and brightness is suspected for SeaLake and Forest; a grayscale re-scoring would test this.

**Limitations:** 64 x 64 patches from European Sentinel-2 scenes only; a single sensor with a mixture of seasons; overlapping tiles may inflate the test score; every image is its own group. Not for legal land-title, tax or subsidy decisions, or for regions outside Europe without re-validation.

## Responsibility questions

1. **Who would be harmed, and how?** Landowners and farmers could be harmed if a wrong label (for example "Industrial" or "Residential" instead of "Pasture" or "AnnualCrop") were used automatically for zoning, tax or subsidy decisions. Planners could also miss real land-use change if a class such as Pasture (recall 0.906) is under-detected.
2. **Where was the model not validated?** On regions outside Europe, on other seasons, sensors and image sizes than 64 x 64, and on a truly separate geographic area. The test set came from the same pool of tiles, and near-neighbours of test tiles are present in training.
3. **What is the human role?** A person reviews low-confidence and high-impact predictions and makes the final decision. The model only proposes labels, and its per-class recall (0.906 to 0.990) shows that some errors are unavoidable.
4. **What evidence would change my conclusions?** A geographically separate test set with clearly lower macro-F1 would show that 0.960 is inflated by neighbouring tiles. A grayscale or colour-shuffle test that collapses macro-F1 would show that the model depends mostly on colour. A Grad-CAM or masking experiment showing that the highlighted regions do not drive the prediction would change my "no shortcut" verdict.
<img width="1284" height="809" alt="image" src="https://github.com/user-attachments/assets/60184512-c78a-4efc-b50f-0527d3b87691" />
<img width="745" height="1600" alt="image" src="https://github.com/user-attachments/assets/e88e4e4b-a5af-4834-9595-417c0f37c924" />
<img width="861" height="848" alt="image" src="https://github.com/user-attachments/assets/901d60eb-1ed7-44c8-a557-3adaa63ac786" />


