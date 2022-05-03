# Standard Library
import re
import time
import logging
import platform
from typing import *
from pathlib import Path
from attr import dataclass

# Third-Party Library
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

init(autoreset=True)


class DatasetPath:
    class PascalVoc2012:
        train_split: List[str]
        val_split: List[str]
        image_folder: Path
        target_folder: Path

        if platform.system() == "Windows":
            base: Path = Path(__file__).resolve().parent.joinpath(
                "datasets", "PascalVoc2012-windows").resolve()
        else:
            assert NotImplementedError, f"{Fore.RED}Please implement the base folder of the Pascal VOC 2012 root"\
                                        f" path in your system"

        image_folder: Path = base.joinpath("ImageSets", "JPEGImages")
        target_folder: Path = base.joinpath("ImageSets", "SegmentationClass")

        def __init__(self, split: str):
            assert split.lower() in [
                "train", 'val'], f"{Fore.RED}Invalid dataset split"
            with self.base.joinpath("ImageSets", "Segmentation", "train.txt").open(mode="r") as f:
                self.train_split: List[str] = [i.strip()
                                               for i in f.readlines()]

            with self.base.joinpath("ImageSets", "Segmentation", "val.txt").open(mode="r") as f:
                self.val_split: List[str] = [i.strip() for i in f.readlines()]


@dataclass
class ProjectPath:
    base: Path = Path(__file__).resolve().parent
    dataset: Path = base.joinpath("datasets")
    log: Path = base.joinpath("log")
    runs: Path = base.joinpath("runs")
    checkpoints: Path = base.joinpath("checkpoints")

    def __attrs_post_init__(self):
        try:
            for k, v in self.__dict__.items():
                if isinstance(v, Path):
                    v.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"{Fore.RED}Initialize project failed: {e}")
            raise e


class Logger(logging.Logger):
    def __init__(self, level: str = "info") -> None:
        super().__init__(name="FCN-Logger", level=getattr(logging, level.upper()))
        self.log_path = []

    def close(self):
        print(f"Shutdown logger, waiting...")
        time.sleep(3)
        logging.shutdown()

        # clean colored output
        for i in range(len(self.log_path)):
            with self.log_path[i].open(mode="r+") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    lines[i] = re.sub(r"..\d\dm", "", lines[i])
                f.seek(0)
                f.write("".join(lines[:-1]))

    def to_file(self, path: Path) -> "Logger":
        self.log_path.append(path if isinstance(path, Path) else Path(path))
        self.log_path[-1].parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=str(self.log_path[-1]))
        self.addHandler(file_handler)
        return self

    def to_terminal(self) -> "Logger":
        import sys
        terminal_handler = logging.StreamHandler(sys.stdout)
        self.addHandler(terminal_handler)
        return self


ImageType = TypeVar("ImageType", np.ndarray, List[np.ndarray], None)
TitleType = TypeVar("TitleType", str, List[str], None)


def get_image(return_png: bool = False, driver: str = "ndarray"):
    assert driver in ["pil", "ndarray"]

    def visualize_func_decider(show_func: Callable = visualize) -> Callable:
        def show_with_png(*args, **kwargs):
            show_func(*args, **kwargs)
            import matplotlib.backends.backend_agg as bagg
            canvas = bagg.FigureCanvasAgg(plt.gcf())
            canvas.draw()
            png, (width, height) = canvas.print_to_buffer()
            png = np.frombuffer(png, dtype=np.uint8).reshape(
                (height, width, 4))

            if driver == 'pil':
                return Image.fromarray(png)
            else:
                return png

        if return_png:
            return show_with_png
        else:
            return show_func

    return visualize_func_decider


@overload
def visualize(image: np.ndarray, title: Optional[Union[str, List[str]]] = None) -> None:
    ...


@overload
def visualize(image: List[np.ndarray], title: Optional[Union[str, List[str]]] = None) -> None:
    ...


# @get_image(return_png=True, driver="pil")
# @get_image(return_png=True, driver="ndarray")
def visualize(image: ImageType, title: TitleType = None) -> None:
    # make images
    assert image is not None, f"image cannot be None"
    if isinstance(image, np.ndarray):
        assert image.ndim in [3, 4], f"Wrong Shape, should be [channel, width, height]" \
                                     f" or [image, channel, width, height], but you offered {image.shape}"
        assert image.shape[-3] in [1, 3], f"Wrong channel, should be [channel, width, height]" \
                                          f" or [image, channel, width, height], but you offered {image.shape}"
        if image.ndim == 3:
            image = [image]
        elif image.ndim == 4:
            image = [image[i, ...] for i in range(len(image))]
    else:
        for i in image:
            assert i.ndim == 3, f"Wrong shape, should be List[np.ndarray[channel, width, height]]," \
                                f" but you offered {i.shape}"
            assert i.shape[0] == 3, f"Wrong shape, should be List[np.ndarray[channel, width, height]]" \
                                    f" but you offered {i.shape}"

    # make title
    assert isinstance(
        title, (str, list)) or title is None, f"Wrong type, should be str or List[str]"
    if isinstance(title, str) or title is None:
        if len(image) > 1:
            repeats = len(image)
        else:
            repeats = 1
        title = [title if isinstance(title, str) else ""] * repeats
    elif isinstance(title, list):
        for i in title:
            assert isinstance(
                i, str), f"{Fore.RED}Wrong type, should be List[str], but you offered {type(title)}"

    import math
    grid_r = int(math.sqrt(len(image)))
    grid_c = math.ceil(len(image) / grid_r)

    fig, ax = plt.subplots(nrows=grid_r, ncols=grid_c, figsize=(
        3 * grid_r, 4 * grid_c), layout="tight")
    if len(image) == 1:
        ax = [[ax]]
    elif grid_r == 1:
        ax = [ax]

    from matplotlib.axes import Axes
    ax: Union[List[Axes], List[List[Axes]]]
    for idx in range(len(image)):
        row = idx // grid_r
        col = idx - row * grid_r
        ax[row][col].imshow(image[idx].transpose(1, 2, 0))
        ax[row][col].set_title(title[idx])


if __name__ == "__main__":
    # test visualize and get_image
    # single_image = np.random.random(size=(3, 256, 256))
    # multiple_image = np.random.random(size=(16, 3, 256, 256))
    # image_list = [np.random.random(size=(3, 256, 256)) for i in range(16)]

    # visualize(multiple_image, [f"random_{i}" for i in range(len(multiple_image))])
    # plt.show()

    # img = get_image(return_png=True, driver="pil")(visualize)(multiple_image,
    #                                                           [f"random_{i}" for i in range(len(multiple_image))])
    # img.show()

    # test project path
    # import pprint
    #
    # project_path = ProjectPath()
    # pprint.pprint(project_path, width=1)

    # test dataset path
    # print(DatasetPath.PascalVoc2012(split="train").train_split)

    # test logger
    logger = Logger().to_file("test1.log").to_file("test2.log").to_terminal()
    logger.info(f"{Fore.RED}Test")
    logger.close()
