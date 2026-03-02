# Fix WSL Installation

Your WSL installation is corrupted. Here are the steps to fix it:

## Option 1: Repair WSL (Quick)

1. Open PowerShell as Administrator
2. Run:
```powershell
wsl --update
wsl --shutdown
```

3. If that doesn't work, unregister and reinstall:
```powershell
wsl --unregister Ubuntu  # or your distro name
wsl --install -d Ubuntu
```

## Option 2: Complete Reinstall (If repair fails)

### Method A: Using wsl --uninstall (Windows 11 22H2+)

1. Open PowerShell as Administrator
2. Uninstall WSL completely:
```powershell
wsl --uninstall
```

3. Restart your computer

4. After restart, reinstall WSL:
```powershell
wsl --install -d Ubuntu
```

5. Restart again if prompted

### Method B: Manual Uninstall (If wsl --uninstall doesn't work)

1. Open PowerShell as Administrator
2. List all installed distributions:
```powershell
wsl --list --verbose
```

3. Unregister each distribution (replace `Ubuntu` with your distro name):
```powershell
wsl --unregister Ubuntu
wsl --unregister Ubuntu-22.04
# Repeat for all listed distros
```

4. Uninstall WSL kernel update:
   - Open "Apps & Features" (Settings → Apps → Apps & features)
   - Search for "Windows Subsystem for Linux Update"
   - Click Uninstall

5. Disable WSL features:
```powershell
dism.exe /online /disable-feature /featurename:Microsoft-Windows-Subsystem-Linux /norestart
dism.exe /online /disable-feature /featurename:VirtualMachinePlatform /norestart
```

6. Restart your computer

7. After restart, enable WSL features:
```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

8. Restart again

9. Install WSL:
```powershell
wsl --install -d Ubuntu
```

10. Set WSL 2 as default:
```powershell
wsl --set-default-version 2
```

## Option 3: Use Windows Features (If commands fail)

1. Open "Turn Windows features on or off"
2. Uncheck "Windows Subsystem for Linux" and "Virtual Machine Platform"
3. Click OK and restart
4. After restart, check both features again
5. Restart again
6. Open PowerShell as Administrator and run:
```powershell
wsl --install -d Ubuntu
```

## After WSL is Fixed

Once WSL is working, run:
```bash
cd /mnt/d/Mamba\ v2
bash setupenv.sh
```

This will create the venv with mamba_ssm support.

## Alternative: Use VPS/Cloud

If WSL continues to have issues, you can:
1. Use the VPS setup script directly on a Linux machine
2. Use Google Colab with GPU
3. Use a cloud VM (AWS, GCP, Azure) with GPU

The training scripts are designed to work on any Linux system with CUDA.
