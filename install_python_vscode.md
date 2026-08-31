# Comprehensive Guide: Installing Python (via Miniconda) & VS Code on Windows and Linux

This guide provides step-by-step instructions for setting up a clean, scalable Python development environment using **Miniconda** (a lightweight distribution of Anaconda) and **Visual Studio Code (VS Code)** on both **Windows** and **Linux** operating systems.

---

## Table of Contents
1. [Overview & Prerequisites](#overview--prerequisites)
2. [Installing Miniconda & VS Code on Windows](#installing-miniconda--vs-code-on-windows)
   - [Step 1: Install Miniconda](#step-1-install-miniconda)
   - [Step 2: Install Visual Studio Code](#step-2-install-visual-studio-code)
   - [Step 3: Verify & Configure Windows Environment](#step-3-verify--configure-windows-environment)
3. [Installing Miniconda & VS Code on Linux](#installing-miniconda--vs-code-on-linux)
   - [Step 1: Install Miniconda](#step-1-install-miniconda-1)
   - [Step 2: Install Visual Studio Code](#step-2-install-visual-studio-code-1)
   - [Step 3: Verify & Configure Linux Environment](#step-3-verify--configure-linux-environment)
4. [Setting Up Python & Miniconda in VS Code](#setting-up-python--miniconda-in-vs-code)
   - [Installing Essential Extensions](#installing-essential-extensions)
   - [Selecting the Conda Python Interpreter](#selecting-the-conda-python-interpreter)
5. [Conda Quick Reference Guide](#conda-quick-reference-guide)

---

## Overview & Prerequisites

### Why Miniconda?
Miniconda is a minimal installer for Conda. It includes only Conda, Python, their dependencies, and a small number of useful packages (such as `pip` and `zlib`). Unlike Anaconda, which pre-installs over 1,500 scientific packages, Miniconda allows you to keep your system lean and install only what you need.

### Why VS Code?
Visual Studio Code is a lightweight, powerful code editor supporting extensive customization, cross-platform usage, integrated debugging, Git integration, and native Jupyter Notebook execution.

---

## Installing Miniconda & VS Code on Windows

### Step 1: Install Miniconda

1. **Download the Installer:**
   - Go to the official Miniconda page: [https://docs.anaconda.com/miniconda/](https://docs.anaconda.com/miniconda/)
   - Download the latest **64-bit Windows Installer** executable (`Miniconda3-latest-Windows-x86_64.exe`).

2. **Run the Installer:**
   - Double-click the downloaded `.exe` file.
   - Click **Next** and accept the License Agreement (**I Agree**).
   - Select installation type: **Just Me** (recommended for standard use).
   - Choose the installation path (default is usually `C:\Users\<YourUsername>\miniconda3`).

3. **Advanced Installation Options:**
   - **Create shortcuts:** Checked (recommended).
   - **Add Miniconda3 to my PATH environment variable:** *Unchecked* (recommended by Anaconda to prevent system-wide conflicts). Use **Anaconda Prompt** or integrate with PowerShell/Command Prompt via `conda init`.
   - **Register Miniconda3 as my default Python 3.x:** Checked.

4. **Finish Setup:**
   - Click **Install**, and once completed, click **Finish**.

Alternatively you can check this [page](https://www.anaconda.com/docs/getting-started/miniconda/install/windows-cli-install#command-prompt) for more information on installing Miniconda via command line.

---

### Step 2: Install Visual Studio Code

1. **Download VS Code:**
   - Visit the official VS Code download page: [https://code.visualstudio.com/](https://code.visualstudio.com/)
   - Download the **Windows x64 User Installer**.

2. **Run the Installer:**
   - Launch the `.exe` setup file and accept the license agreement.
   - Choose the default installation location.
   - In the **Select Additional Tasks** step, check the following options:
     - [x] Add "Open with Code" action to Windows Explorer file context menu
     - [x] Add "Open with Code" action to Windows Explorer directory context menu
     - [x] Register Code as an editor for supported file types
     - [x] Add to PATH (requires shell restart)

3. **Complete Installation:**
   - Click **Install** and then **Finish**.

---

### Step 3: Verify & Configure Windows Environment

1. Search for **Anaconda Prompt (miniconda3)** in the Windows Start menu and launch it.
2. Verify Conda installation by checking its version:
   ```cmd
   conda --version
   ```
3. Initialize Conda for standard Windows PowerShell or Command Prompt:
   ```cmd
   conda init powershell
   ```
4. Restart your PowerShell terminal. You should now see `(base)` before your command prompt line, indicating the default Conda environment is active.

---

## Installing Miniconda & VS Code on Linux

### Step 1: Install Miniconda

1. **Open Terminal** (`Ctrl + Alt + T`).

2. **Download the Linux Installer Script:**
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

3. **Run the Installer Script:**
   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

4. **Follow the On-Screen Prompts:**
   - Press `Enter` to review the license agreement, scroll using `Space`, and type `yes` to accept.
   - Confirm the default installation directory (usually `~/miniconda3`).
   - When prompted with *"Do you wish the installer to initialize Miniconda3 by running conda init?"*, type `yes`.

5. **Apply Changes:**
   - Reload your shell profile:
     ```bash
     source ~/.bashrc
     ```
   - (Optional) Clean up the installer script:
     ```bash
     rm Miniconda3-latest-Linux-x86_64.sh
     ```

---

### Step 2: Install Visual Studio Code

#### Option A: Ubuntu / Debian (`.deb` Package)

1. Download the `.deb` package:
   ```bash
   wget -O vscode.deb https://go.microsoft.com/fwlink/?LinkID=760868
   ```
2. Install via `apt`:
   ```bash
   sudo apt update
   sudo apt install -y ./vscode.deb
   rm vscode.deb
   ```

#### Option B: Fedora / RHEL / CentOS (`.rpm` Package)

1. Import the Microsoft GPG key and repository:
   ```bash
   sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
   sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" > /etc/yum.repos.d/vscode.repo'
   ```
2. Install VS Code:
   ```bash
   sudo dnf check-update
   sudo dnf install code
   ```

#### Option C: Snap Package (Universal Linux)

```bash
sudo snap install code --classic
```

---

### Step 3: Verify & Configure Linux Environment

1. Verify Conda installation:
   ```bash
   conda --version
   ```
2. Verify VS Code installation:
   ```bash
   code --version
   ```

---

## Setting Up Python & Miniconda in VS Code

### Installing Essential Extensions

1. Open **VS Code**.
2. Go to the Extensions view (`Ctrl + Shift + X`).
3. Search for and install the following extensions:
   - **Python** (by Microsoft) — Provides auto-completion, linting, debugging, and environment selection.
   - **Pylance** (by Microsoft) — High-performance language server for Python.
   - **Jupyter** (by Microsoft) — Enables execution of Jupyter Notebooks directly inside VS Code.

---

### Selecting the Conda Python Interpreter

1. Open or create a Python file (`script.py`) or folder in VS Code.
2. Open the Command Palette using `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS).
3. Type and select **Python: Select Interpreter**.
4. A dropdown list will appear showing detected Python environments. Choose your Miniconda environment (e.g., `Python 3.x.x ('base': conda)` or custom conda environment path).
5. Open an integrated terminal in VS Code (`Ctrl + ~`). You should automatically see the Conda environment activated at the prompt prompt bar.

---

## Conda Quick Reference Guide

Below are essential commands for managing Python environments with Conda:

| Task | Command |
| :--- | :--- |
| **Create Environment** | `conda create --name myenv python=3.11` |
| **Activate Environment** | `conda activate myenv` |
| **Deactivate Environment** | `conda deactivate` |
| **List Environments** | `conda env list` |
| **Install Packages** | `conda install numpy scipy pandas` |
| **Install via Pip in Conda** | `pip install package_name` |
| **List Installed Packages** | `conda list` |
| **Remove Environment** | `conda env remove --name myenv` |

---

*End of Guide.*
