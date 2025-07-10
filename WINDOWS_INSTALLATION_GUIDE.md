# 🪟 Windows Installation Guide - Arjay Siega

## For Python 3.11 on Windows

### 🚀 **Step-by-Step Installation**

#### 1. First, upgrade pip (recommended)
```cmd
python -m pip install --upgrade pip
```

#### 2. Install the MT5 requirements
```cmd
pip install -r requirements_mt5.txt
```

If you get any errors, try installing packages individually:

#### 3. Core packages first
```cmd
pip install numpy pandas scipy matplotlib seaborn
```

#### 4. GARCH modeling (most important)
```cmd
pip install arch
```

#### 5. MT5 integration
```cmd
pip install MetaTrader5
```

#### 6. Configuration and utilities
```cmd
pip install PyYAML python-dotenv structlog
```

#### 7. Optional ML packages
```cmd
pip install scikit-learn joblib
```

### 🔧 **If You Still Get Errors**

#### Alternative Installation Method
```cmd
# Install with no dependencies first, then add them
pip install --no-deps arch
pip install numpy pandas scipy

# Then install arch properly
pip install arch --upgrade
```

#### Or use conda (if you have Anaconda/Miniconda)
```cmd
conda install numpy pandas scipy matplotlib seaborn
conda install -c conda-forge arch
pip install MetaTrader5 PyYAML python-dotenv structlog
```

### ✅ **Test Your Installation**

Run this command to test everything works:
```cmd
python -c "import numpy, pandas, arch, MetaTrader5; print('✅ All packages installed successfully!')"
```

### 🎯 **Your Account is Ready**

Once packages are installed, you can start trading:

```cmd
cd F:\Intraday-strategy-using-GARCH-Model-main
python src/main.py
```

### 📋 **What Each Package Does**

- **numpy/pandas**: Data processing for market data
- **scipy**: Mathematical functions for statistics  
- **arch**: GARCH volatility modeling (core algorithm)
- **MetaTrader5**: Connection to your MT5 terminal
- **matplotlib/seaborn**: Charts and visualization
- **PyYAML**: Configuration file reading
- **structlog**: Advanced logging system
- **scikit-learn**: Machine learning features

### 🚨 **Troubleshooting Common Issues**

#### Error: "Microsoft Visual C++ 14.0 is required"
**Solution**: Install Microsoft C++ Build Tools
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "C++ build tools" workload
3. Restart and try again

#### Error: "No module named 'MetaTrader5'"
**Solution**: 
```cmd
pip install MetaTrader5 --upgrade
```

#### Error: "arch package not found"
**Solution**: 
```cmd
pip install arch --no-cache-dir
```

#### Error: "Cannot find MT5 terminal"
**Solution**: Make sure MT5 is installed at:
```
C:\Users\guest_1\AppData\Roaming\MetaTrader 5\terminal64.exe
```

### 🎉 **Ready to Trade!**

Once everything is installed:

1. **Make sure MT5 terminal is running**
2. **Login to your demo account (94435704)**  
3. **Run the GARCH strategy**:
   ```cmd
   python src/main.py
   ```

### 📊 **Expected Output**
```
[INFO] Trading engine initialized
[INFO] Connected to MT5 - Account: 94435704, Balance: $10,000.00
[INFO] GARCH models fitted successfully for EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD
[INFO] Trading engine started successfully
```

### 🔍 **Monitor Your Trading**
```cmd
# In a new command prompt, watch the logs
type logs\trading_*.log
```

**You're all set for automated GARCH trading on Windows! 🎯📈**