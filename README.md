# SXD-TPA: α-Decay Half-Life Calculation and BNN Neural Network Prediction Program

Based on the TPA, this program calculates α decay half-lives. Combined with a BNN (Bayesian Neural Network) to extract the preformation factor Pα, it provides a complete workflow integrating theoretical calculations with machine-learning predictions.

---

## 1. File Description

    ### File 1: SXD-TPA.exe (Executable File)

    -**Function**: Calculates the theoretical half-life T^{cal} based on Pα = 1
     - **Input File**: `input535.txt`
    - **Output File**: `output535.txt`
    - **Subsequent Use**: After execution, use Equation (6) from the theoretical framework to extract the     experimental Pα^{exp} data

    ### File 2: codeBNN.py (BNN Neural Network Program)

    -**Function**: Trains/runs the BNN to output the Pα^{BNN-Auto} data
    - **Input File**: `BNN_input2.txt`
    - **Output File**: `BNN_output2.txt` (containing Pα^{BNN-Auto})

    ### File 3: SXD-TPA.exe (Executable File, second invocation)

    -**Function**: Uses the Pα^{BNN-Auto} data obtained from File 2 to calculate the half-life T^{BNN-Auto} within     the theoretical model framework
    - **Input File**: `BNN_input.txt`
    - **Output File**: `BNN_output.txt`

    ### File 4: SXD-TPA.exe (Executable File, third invocation)

    -**Function**: Calculates the predicted data T^{BNN-Auto_Pre} under the theoretical model
    - **Input File**: `BNN-Pre_input.txt`
    - **Output File**: `BNN-Pre_output.txt`

---

## 2. Execution Order

    `` run1 → run2 → run3 → run4 ``

    ### Detailed Workflow:

    | Step   |    Program     |      Input File             |           Output File       |           Purpose                                           |
    |           |                      |                                 |                                    |                                                                  |
    | run1   | SXD-TPA.exe | input535.txt              | output535.txt              | Calculate T^{cal} (Pα = 1)                           |
    | run2   | run_bnn_auto.py  | BNN_input2.txt   | BNN_output2.xlsx       | Calculate Pα^{BNN-Auto}                          |
    | run3   | SXD-TPA.exe | BNN_input.txt           | BNN_output.txt           | Calculate T^{BNN-Auto}                            |
    | run4   | SXD-TPA.exe | BNN-Pre_input.txt    | BNN-Pre_output.txt     | Calculate predicted T^{BNN-Auto_Pre}       |

---

## 3. Notes

    (1). All input files must be placed in the same directory as the executable
    (2). If the Q value is too low and T1/2 = Infinity, it indicates that the nuclide is extremely stable under the current              model (physically reasonable)
    (3). The execution order must not be changed: run1 → run2 → run3 → run4

## 4. Install

Use Python 3.9 or a compatible environment:

```powershell
python -m pip install -r .\requirements.txt
```

Place `BNN_input2.xlsx` at `.\data\BNN_input2.xlsx`, or pass an explicit `--data-path`.

## 5. Run the released BNN-Auto model

From this release directory, the following PowerShell command reproduces the BNN-Auto confirmation protocol. The explicit data path below uses the working data file outside this release directory.

```powershell
python.exe run_bnn_auto.py --data-path "BNN_input2.xlsx" --output-dir .\outputs --device cpu --models all
```

 To run all included comparison methods, replace `--models BNN-Auto` with `--models all`.
