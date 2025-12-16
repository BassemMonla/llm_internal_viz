import os

path = r"C:\Users\BassemMonla\.lmstudio\models\lmstudio-community\gpt-oss-20b-GGUF"
print(f"Checking path: {path}")

if os.path.exists(path):
    print("Path exists!")
    if os.path.isdir(path):
        print("It is a directory. Contents:")
        try:
            for f in os.listdir(path):
                print(f" - {f}")
        except Exception as e:
            print(f"Error listing dir: {e}")
    else:
        print("It is a file.")
else:
    print("Path does NOT exist.")
    parent = os.path.dirname(path)
    print(f"Checking parent: {parent}")
    if os.path.exists(parent):
        print("Parent exists. Contents:")
        try:
             for f in os.listdir(parent):
                print(f" - {f}")
        except Exception as e:
            print(f"Error listing parent: {e}")
    else:
        print("Parent does NOT exist.")
