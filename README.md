# 🏨 Hotel Booking Data Cleaning & Preprocessing

This project focuses on cleaning and preprocessing a hotel bookings dataset (`hotel_bookings.csv`) to prepare it for a machine learning model that predicts last-minute booking cancellations.

The quality of the data preprocessing directly impacts the accuracy and reliability of future predictive models.

## 📂 Dataset

**Source:** Property Management System (PMS) of a hotel  
**File:** `hotel_bookings.csv`  
**Rows:** ~119,390 bookings  
**Columns:** 31 features including numeric, categorical, and date/time fields.

### Key Columns:
- `is_canceled` – **Target column**: `0` = not canceled, `1` = canceled
- `adr` – Average daily rate per booking
- `lead_time` – Days between booking and arrival
- `stays_in_weekend_nights` & `stays_in_week_nights` – Number of nights booked
- `adults`, `children`, `babies` – Number of guests
- `meal`, `market_segment`, `country` – Categorical features
- `reservation_status` – Final booking status *(Removed to prevent data leakage)*

## 🛠️ Project Phases

### Phase 1: Data Quality Analysis
- **Load & Inspect Data**: Used `.info()`, `.describe()`, and head/tail exploration.
- **Identify Missing Values**: Visualized using heatmaps or `missingno` matrix.
- **Detect Outliers**:
  - Boxplots for numeric features (`adr`, `lead_time`, etc.)
  - IQR method to identify extreme values.
- **Initial Observations**:
  - Some bookings had a Check-Out status but `total_nights = 0` → inconsistent.
  - Extreme `adr` values (very low or >1000) → outliers.

### Phase 2: Data Cleaning
- **Handle Missing Values**:
  - `company` & `agent` → Replaced with `"None"` or `0`.
  - `country` → Replaced missing with mode or `"Unknown"`.
  - `children` → Imputed with median/mode.
- **Remove Duplicates**: `df.duplicated().sum()` → Dropped exact duplicates.
- **Handle Outliers**:
  - Capped `adr` values above 1000 to 1000.
  - Optional: Applied a floor for minimum values.
- **Fix Data Types**:
  - Date columns converted to `datetime`.
  - Boolean columns created for binary flags.

### Phase 3: Feature Engineering & Preprocessing
- **New Features**:
  - `total_guests` = `adults` + `children` + `babies`
  - `total_nights` = `stays_in_weekend_nights` + `stays_in_week_nights`
  - `is_family` = `True/False` if `children` or `babies` > 0
- **Encoding Categorical Variables**:
  - Low-cardinality (e.g., `meal`, `market_segment`) → One-Hot Encoding.
  - High-cardinality (e.g., `country`) → Frequency Encoding or grouped rare categories into `"Other"`.
- **Avoid Data Leakage**:
  - Dropped `reservation_status` & `reservation_status_date` as they contain outcome information.

### Phase 4: Train/Test Split
- Split dataset into training and testing sets:
  - `test_size = 0.2` (20% test data)
  - `random_state = 42` (for reproducible results)
- Separated features (`X`) and target (`y` = `is_canceled`).

## ⚡ Key Notes
- `is_canceled = 1` → Booking was canceled.
- `is_canceled = 0` → Booking was not canceled (guest stayed).
- Check-Out bookings with `total_nights = 0` → Removed as inconsistent data.
- Outliers in `adr` were capped to prevent skewed models.
- Rare categories in high-cardinality columns were grouped to `"Other"` to avoid creating too many sparse features.

## 📈 Example Insights
- Most bookings come from **Online TA** and the **top 5 countries** (excluding "Other") dominate the dataset.
- **ADR** has a long-tail distribution, with most values between **50–200** and few extreme outliers.
- **Families** (`children`/`babies` > 0) represent a smaller portion of bookings.

## 🛠️ Tools & Libraries
- **Python 3.x**
- **Pandas** → Data manipulation
- **NumPy** → Numeric calculations
- **Seaborn** / **Matplotlib** → Data visualization
- **Scikit-learn** → Train/test split, preprocessing

---

**Note:** This cleaned dataset is now ready for exploratory data analysis (EDA) and building a predictive model for booking cancellations.
