import argparse
import zipfile
import pathlib
import warnings


class TeamMember:
    def __init__(self, first_names, last_names):
        self.first_names = first_names
        self.last_names = last_names

    def __str__(self):
        # lastname1-firstname1_lastname2-firstname2
        return "-".join(self.last_names + self.first_names).lower()
        # firstname1-lastname1_firstname2-lastname2
        # return "-".join(self.first_names + self.last_names).lower()

    def __lt__(self, other):
        # define sorting order for TeamMembers
        return "".join(self.last_names) < "".join(other.last_names)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def yesno(msg, default=None):
    valid = {
        # lowercase only
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "": default,
    }

    answer = None

    msg += " [y/n]" + (f" (default: {default})" if default else "") + ": "

    while answer is None:
        answer = input(msg)
        answer = valid.get(answer)

    return answer


def find_notebooks(path: pathlib.Path, ignore: str = None):
    # autodetect the notebook files
    if ignore is None:
        ignore = set()
    else:
        ignore = set(path.glob(ignore))

    ccommented = set(path.glob("*-cross-commented.ipynb")) - ignore
    commented = set(path.glob("*-commented.ipynb")) - ccommented - ignore
    notebook = set(path.glob("*.ipynb")) - commented - ccommented - ignore

    # exception handling
    if len(notebook) > 1:
        raise RuntimeError(f"Found more than one main notebook in {path}")

    if not notebook:
        raise FileNotFoundError(f"Found no main notebook in {path}")
    else:
        notebook = notebook.pop()
        print(f"Found main notebook {notebook}")

    if len(commented) > 1:
        raise RuntimeError(f"Found more than one commented notebook in {path}")

    if not commented:
        warnings.warn(f"There is no commented notebook in {path}.")
        if not yesno(f"Are you sure you want to continue?"):
            raise FileNotFoundError

        commented = None
    else:
        commented = commented.pop()
        print(f"Found commented notebook {commented}")

    if len(ccommented) > 1:
        raise RuntimeError(f"Found more than one cross-commented notebook in {path}")

    if not ccommented:
        ccommented = None
    else:
        ccommented = ccommented.pop()
        print(f"Found cross-commented notebook {ccommented}")

    # return all available notebooks and ignored files
    return notebook, commented, ccommented, ignore


def convert_ipynb_html(path):
    import subprocess
    # we use jupyter for the conversion
    subprocess.run(["jupyter", "nbconvert", "--to", "html", str(path)])
    # return the path to the new html file
    return path.with_suffix(".html")


def main():
    # members must be sorted
    members = sorted([
        TeamMember(["Nicolas"], ["Wolf"]),
        TeamMember(["Lars", "Erik"], ["Kuehmichel"]),
    ])

    # members string where individual members are separated by _
    members = "_".join([str(m) for m in members])

    if not is_ascii(members):
        warnings.warn("Members contain unicode characters, which may violate the naming convention.")

    parser = argparse.ArgumentParser(description="Automatically create the hand-in file for the lecture FML.")
    parser.add_argument("name", type=str,
                        help="The name of the exercise, to be included at the end of the zip filename.")
    parser.add_argument("-m", "--more", type=str,
                        help="Glob Pattern to determine additional files which should be added to the Hand-In.")
    parser.add_argument("--ignore", type=str,
                        help="Glob Pattern to determine which files should be ignored. "
                             "This takes priority over all other file detections.")

    args = parser.parse_args()

    # check if directory exists
    ex_path = pathlib.Path(args.name)
    if not ex_path.is_dir():
        raise ValueError(f"Exercise Path '{args.name}' does not exist.")

    # first check for notebooks before creating zip file
    nb, com, ccom, ignore = find_notebooks(ex_path, args.ignore)

    zip_path = ex_path / (members + "_" + args.name + ".zip")

    # create the zip hand-in file
    zipf = zipfile.ZipFile(zip_path, "w")

    # notebook and directory are fine, convert the notebook to html
    nb_html = convert_ipynb_html(nb)
    # add the html and notebook to the zip
    zipf.write(nb, nb.name)
    zipf.write(nb_html, nb_html.name)

    if com:
        com_html = convert_ipynb_html(com)
        zipf.write(com, com.name)
        zipf.write(com_html, com_html.name)

    if ccom:
        ccom_html = convert_ipynb_html(ccom)
        zipf.write(ccom, ccom.name)
        zipf.write(ccom_html, ccom_html.name)

    if args.more:
        files = set(ex_path.glob(args.more)) - ignore
        print(f"adding {len(files)} more files...")

        for file in ex_path.glob(args.more):
            # add the file without adding the ex_path folder in
            zipf.write(file, pathlib.Path(*file.parts[1:]))

    zipf.close()

    return 0


if __name__ == "__main__":
    main()
