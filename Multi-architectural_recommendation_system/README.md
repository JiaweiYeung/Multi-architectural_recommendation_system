## Architecture Style Retrieval System

## 🛠️ How to Use It

### ⚠️If you download it and run it directly, make sure it's on macOS!

### We haven't tested if it works on windows, you need to replace all the paths with ones that work on windows.

1. **Upload an Image**
    Drag and drop or select a building image to upload via the main interface. Once uploaded, the system will automatically classify the architectural style of the image and display the prediction along with a confidence score.

2. **View Top-10 Local Similar Images**
    In the **Top 10 Local Similar Images** section, you can explore visually similar images retrieved from the local dataset. Each image includes its predicted style, similarity score, and a download link. You can click the ⭐ **Favorite** button to select the one you find most similar.

3. **Retrieve Top-K Similar Online Images**
    In the **Top K Similar Online Images** section, the system performs an online search based on your favorited image. You can also enhance the search by entering additional textual hints in the sidebar, such as:

   - **Location / Style** (e.g., "France", "Gothic")
   - **Time Period** (e.g., "19th century", "night view")
   - **Top K Results** (1–10)

   The retrieved images will include direct source links for further reference.



# Catalogs

├─ dataset
│   └─ data_clean
│       └─ rename_images.py
│
├─ images
│   ├─ bgimage
│   ├─ readme_img
│   ├─ test_image
│   └─ usydlogo.png
│
├─ MR                             # main project 
│   ├─ main.py                    # entrances
│   └─ ui_v4.py                   # Web UI
│
├─ part1_model                  		  # Model Training and Feature Extraction Module
│   ├─ best_grid_50vit_deep.pth  	# model weight
│   ├─ best_hyperparams_deep.json     # best parameter
│   ├─ feature50.py
│   ├─ feature_vit50_new.py
│   ├─ features_50vit.csv         # Extracted features (you can ignore the upload)
│   ├─ trainer.py
│   └─ trainer_deeper.py
│
├─ part2_localsearch              # Local Image Retrieval Module
│   ├─ local_search.py
│   └─ local_top.py
│
├─ part3_onlinesearch             # Online image retrieval module (multimodal)
│   └─ online_multi_mod.py
│
├─ README.md
└─ requirements.txt



# Flow Chart

![images/readme_img/md_image_1.png](images/readme_img/md_image_1.png)

#  Setup

### ⚠️ If you download it and run it directly, make sure it's on macOS!

### We haven't tested if it works on windows, you need to replace all the paths with ones that work on windows.

#### You need to wait for your environment to install the python packages (everything needed is already listed in requirements.txt)

#### When everything is ready, run the main.py file and wait for the default browser to pop up the WebUI.

#### (Everything works as in the video.)

 