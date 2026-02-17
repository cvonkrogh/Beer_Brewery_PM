# Beer_Brewery_PM
 
### Production Management & Demand Forecasting for a Brewery

---

## 📌 Project Overview

This project develops a **demand forecasting and production planning model** for a brewery.  

The objective is to determine:

> **When should we start brewing each beer to meet future demand while minimizing stockouts and overproduction?**

Using historical sales invoice data, the project builds a forecasting pipeline and integrates it with an inventory control model to generate production start recommendations.

---

## 🎯 Business Problem

Brewing beer involves:

- Production lead times (brewing + fermentation)
- Capacity constraints
- Demand variability
- Inventory holding costs
- Stockout risks

Incorrect production timing can result in:

- Lost sales (stockouts)
- Excess inventory
- Increased storage costs
- Inefficient tank utilization

This project aims to provide a **data-driven brewing trigger system**.

---

## 📊 Data

The dataset contains transactional invoice-level sales data including:

- Invoice number
- Invoice date
- Customer information
- Product (beer type)
- Product category
- Units sold
- Total liters sold
- Sales region
- Pricing class

Each row represents one product line within an invoice.

---

## ⚙️ Methodology

The project follows a structured pipeline:

### 1️⃣ Data Aggregation
- Convert invoice-level sales into weekly product demand (liters)
- Clean and standardize date formats
- Handle missing values

### 2️⃣ Demand Forecasting
Models implemented:
- Moving Average (baseline)
- ARIMA
- Prophet (optional advanced model)

Forecast accuracy evaluated using:
- MAE
- RMSE



