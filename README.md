# 🌊 Flood Detection using ResNet50 & UNet++

An end-to-end deep learning system for high-resolution flood mapping and water segmentation. The application uses **ResNet50 + UNet++** to segment satellite imagery into land, permanent water, and flood water in real-time, pulling Sentinel-1 Synthetic Aperture Radar (SAR) data dynamically from Google Earth Engine (GEE).

```
┌─────────────────────────────────┐       API Requests       ┌──────────────────────────────┐
│  Next.js Map UI (Port 3000)     │ ───────────────────────> │  FastAPI Backend (Port 8000) │
│                                 │                          │                              │
│  - MapLibre GL Interactive Map  │ <─────────────────────── │  - ResNet50 + UNet++ Model   │
│  - Coordinates & Shapefile Tabs │     GeoJSON + GeoTIFF    │  - Google Earth Engine API   │
│  - Downloader (TIF / SHP.ZIP)   │                          │  - Rasterio & Geopandas I/O  │
└─────────────────────────────────┘                          └──────────────────────────────┘
```

---

## 🔬 The Science & Machine Learning (How It Works)

### 1. Dynamic Data Acquisition (Sentinel-1 SAR)
Optical satellites (like Sentinel-2 or Landsat) are heavily limited during floods because flood events are almost always accompanied by cloud cover. 
To bypass this, this system utilizes **Sentinel-1 Synthetic Aperture Radar (SAR)**:
*   **Active Sensor:** Sends radar pulses to the Earth and measures the backscatter (reflection).
*   **All-Weather, Day & Night:** Radar waves penetrate cloud cover, rain, and darkness.
*   **Water Properties:** Smooth water acts like a mirror to radar pulses, reflecting them away from the satellite, which appears as very low backscatter (dark pixels) in the VV and VH bands, making it highly distinguishable from rough land surface.

### 2. Neural Network Architecture (ResNet50 + UNet++)
*   **Encoder (ResNet50):** A deep residual network pre-trained on ImageNet that extracts robust multi-scale geographical features.
*   **Decoder (UNet++):** An advanced nested U-Net architecture. Unlike a standard U-Net, UNet++ uses **nested, dense skip pathways** that bridge the semantic gap between the encoder and decoder feature maps, capturing fine-grained flood boundaries.
*   **Semantic Segmentation (3 Classes):** The model performs pixel-level classification into:
    *   `Class 0`: Dry Land (Non-water)
    *   `Class 1`: Permanent Water Bodies (lakes, rivers, oceans)
    *   `Class 2`: Flood Water (newly inundated areas)

### 3. Sliding-Window Mosaic Inference
Since high-resolution satellite imagery covering an entire province or region is too massive to fit into GPU memory, the backend utilizes a **tiled predictor**:
*   Divides the requested bounding box or shapefile polygon into sliding-window patches.
*   Runs predictions on each patch using the deep learning model.
*   Merges overlapping patches using a soft-voting blending mask to remove seam lines.
*   Outputs a unified, georeferenced `.tif` raster mask and vectorize it into a downloadable `.shp.zip` polygon.

---

## 📁 Repository Layout

The project has been cleaned and organized into a modular structure:

```
BTP-Final-Code/
├── flood-detection-src/       # 🐍 Python: ML Model & Web API
│   ├── api.py                 # FastAPI Web Server entrypoint
│   ├── inference.py           # FloodPredictor class & PyTorch model loader
│   ├── model.py               # ResNet50 + UNet++ PyTorch architecture
│   ├── dataset.py             # Custom tile-generator dataset
│   ├── job_runner.py          # Asynchronous job orchestration for large requests
│   ├── area_calculator.py     # High-precision km² area calculators
│   ├── raster_to_vector.py    # Vectorizer converts predicted raster to shapefile
│   ├── shapefile_handler.py   # Handles user uploaded shapefile unzipping & clipping
│   └── tiled_predictor.py     # Sliding-window mosaic prediction logic
│
├── frontend/                  # ⚛️ Next.js & React: Interactive Web UI
│   ├── app/                   # Next.js App Router (pages, layout, globals.css)
│   ├── components/            # Interactive Map, UI Panels, Inputs & Results
│   ├── lib/                   # API utilities, Mapbox configurations, and typings
│   ├── public/                # Static assets, icons, and sample files
│   │   ├── Coordinates.jpeg   # Sample coordinates reference image
│   │   └── sample-shapefiles/ # Zipped shapefiles for quick upload testing
│   ├── package.json           # Node.js frontend dependencies
│   └── tailwind.config.ts     # Modern CSS design framework config
│
├── Training NoteBooks/        # 📓 Jupyter notebooks for model training & data prep
│   ├── 1-DataSet.ipynb
│   ├── 2-Training.ipynb
│   └── 3-Visualization.ipynb
│
├── checkpoint-v3/             # 🧠 Holds the active trained model (gitignored)
│   └── best_dice.pth          # Active model weight file (130 MB)
│
├── samples/                   # 📁 Testing Samples (Shapefiles & Coordinates)
│   ├── Coordinates.jpeg       # Reference screenshot with bounding box coordinates
│   └── *.zip                  # Zipped sample shapefiles for upload testing (Bolivia, Spain, Sindh)
│
├── .env                       # 🔐 Private keys & environment variables (gitignored)
├── gee-key.json               # 🔐 Google Earth Engine service account key (gitignored)
├── backend-requirements.txt   # Python core libraries (PyTorch, Pyogrio, Rasterio)
├── setup.sh                   # 🛠️ Automated one-time dependency installer
└── run.sh                     # 🚀 Unified one-command server launcher
```

---

## 🔑 Obtaining API Keys & Setup

Before running the application, you need to acquire two access keys: **Google Earth Engine Service Account** (to download satellite imagery) and **ngrok Auth Token** (optional, to make your local server shareable on the web).

### 1. Google Earth Engine (GEE) Service Account Key
To download Sentinel-1 SAR imagery in real-time, the app requires authentication with GEE:
1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  Create a project (or select an existing one).
3.  Search for **Earth Engine API** in the library and click **Enable**.
4.  Navigate to **IAM & Admin > Service Accounts** and click **Create Service Account**.
5.  Give it a name (e.g., `gee-flood-detector`), and create it.
6.  Once created, click on the Service Account email, go to the **Keys** tab, click **Add Key > Create New Key**, select **JSON**, and download it.
7.  Rename the downloaded file to `gee-key.json` and place it in the **root** of this project folder (`BTP-Final-Code/gee-key.json`).

### 2. ngrok Auth Token (Optional)
If you want to share your running application link with a friend or deploy the frontend on Vercel while running the heavy ML backend on your local GPU/laptop:
1.  Sign up for a free account at [ngrok.com](https://ngrok.com/).
2.  Go to your ngrok Dashboard and copy your **Authtoken**.
3.  Add it to your `.env` file as shown below.

### 3. Setting Up the `.env` File
Create a file named `.env` in the **root** of the project (`BTP-Final-Code/.env`) and add the following lines, replacing the values with your own:

```env
# Path to your Google Earth Engine Service Account Key
FLOOD_GEE_KEY="./gee-key.json"

# Your ngrok auth token (optional - for public tunnels)
NGROK_AUTH_TOKEN="your_ngrok_auth_token_here"
```

---

## 🚀 How to Run the App on a New System

Since we have created automated orchestration scripts, setting up and running this dual-server application on a new system is incredibly simple.

### Step 1: Clone the repository and navigate into it
```bash
git clone https://github.com/YOUR_USERNAME/flood-detection-resnet50.git
cd flood-detection-resnet50
```

### Step 2: Add your credentials
1. Place your downloaded service account key as `gee-key.json` in the root folder.
2. Create a `.env` file in the root folder and add your keys (as described in the Key section above).

### Step 3: Run the automated installer
Run the automated `setup.sh` script. This script will check your environment (Python, Node.js), initialize a clean Python virtual environment, download Next.js frontend requirements, and install backend packages:
```bash
# Make script executable
chmod +x setup.sh

# Run installation
./setup.sh
```
> [!TIP]
> **No C-Compilation Needed:** By replacing `fiona` with `pyogrio` in the backend dependencies, the setup script installs a pre-compiled geospatial wheel. You do **not** need to install complex local compilers or C-libraries (like GDAL) on your host operating system!

### Step 4: Run the application
Run the unified launcher script. It will clean up ports, load your `.env` keys, activate your Python virtual environment, and launch **both** the Next.js frontend and FastAPI backend in parallel within a single console tab:
```bash
# Make script executable
chmod +x run.sh

# Start the application
./run.sh
```

Once started, the script will output the running URLs. 
1. Open your browser and go to **`http://localhost:3000`** to interact with the map application!
2. To stop both servers cleanly, simply press **`Ctrl + C`** in your terminal window.

---

## 🧪 Testing the Application with Samples

The `samples/` folder in your project root contains pre-packaged test cases to verify the application is working correctly.

### 1. Bounding Box Test (Coordinates Tab)
To run a prediction by typing in coordinates:
1. Open the [samples/Coordinates.jpeg](file:///Users/yashchawla/Downloads/BTP-Final-Code/samples/Coordinates.jpeg) reference image to see the coordinates of a known flood event in Beni, Bolivia.
2. In the **Coordinates** panel on the web interface, enter the values shown in the image:
   * **Longitude Min:** `-66.0`
   * **Latitude Min:** `-13.7`
   * **Longitude Max:** `-65.95`
   * **Latitude Max:** `-13.65`
3. Enter the flood date: `2018-02-15`.
4. Click **Predict on coordinates** to pull SAR imagery from Google Earth Engine and run the model!

### 2. Custom Polygon Test (Shapefile Tab)
To run a prediction within a custom shapefile boundary:
1. Go to the **Shapefile** tab on the web interface.
2. Drag and drop any `.zip` shapefile from your [samples/](file:///Users/yashchawla/Downloads/BTP-Final-Code/samples/) folder (e.g., `bolivia_irregular.zip` or `spain_coastal.zip`) into the upload dropzone.
3. Enter the corresponding date for the Sentinel-1 image acquisition:
   * For **Bolivia** samples (`bolivia_*.zip`): use `2018-02-15`
   * For **Spain** samples (`spain_coastal.zip`): use `2017-12-29`
   * For **Pakistan** samples (`pakistan_*.zip`): use `2022-09-01`
4. Click **Predict on shapefile**. The server will extract the shapes, crop the fetched radar imagery, mask the ResNet50 predictions to your exact polygon boundaries, and calculate the affected area in km²!
