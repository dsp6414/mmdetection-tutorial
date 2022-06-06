# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
import shutil
import json
from collections import OrderedDict
from typing import IO, Any, Callable, Dict, List, MutableMapping, Optional, Union
import logging
import errno

r"""
函数说明:
    功能：判断文件或目录是否存在
    参数:
        filename：文件或目录路径名
    返回值:
        bool
"""

def is_exists(filename): 
    if isinstance(filename, str):
        if os.path.exists(filename):
            return True     
    return False

r"""
函数说明:
    功能：获取项目主目录
    参数:
        无
    返回值:
        str
"""

def get_cur_dir():
    return os.path.abspath(os.getcwd())

r"""
函数说明:
    功能：获取运行文件目录
    参数:
        无
    返回值:
        str
"""
def get_running_file_dir():
    return Path(os.path.abspath(__file__)).parent

r"""
函数说明:
    功能：获取全路径名
    参数:
        无
    返回值:
        str or None
"""
def get_fullpath_name(file_name):
    if(is_exists(file_name)):
        return os.path.abspath(file_name)
    else:
        return None

r"""
函数说明:
    功能：获取父目录
    参数:
        file_or_dir_name
    返回值:
        str or None
"""
def get_parent_dir(file_or_dir_name):
    if is_exists(file_or_dir_name):
        return os.path.dirname(file_or_dir_name)
    else:
        return None

r"""
函数说明:
    功能：创建目录
    参数:
        无
    返回值:
        str or None
"""
def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


r"""
函数说明:
    功能：文件拷贝
    参数:
        src,des
    返回值:
        无
"""
def file_copy(src,des):
    if is_exists(src):
        shutil.copyfile(src,des)

r"""
函数说明:
    功能：获取文件基本名
    参数:
        file_path_name
    返回值:
        str or None
"""
def get_basename(file_path_name): #从全路径中得到文件名
    if is_exists(file_path_name):
        return os.path.basename(file_path_name)
    else:
        return None

r"""
函数说明:
    功能：获取文件扩展名
    参数:
        file_path_name
    返回值:
        str or None
"""
def get_file_ext_name(file_path_name): #return  (name,ext)
    if is_exists(file_path_name):
        return os.path.splitext(file_path_name)[1]
    else:
        return None

r"""
函数说明:
    功能：获取目录文件列表
    参数:
        dir_name：目录名
    返回值:
        list or None
"""
def get_files(dir_name):
    if is_exists(dir_name):
        files = []
        for f in os.listdir(dir_name):
            fullpath = os.path.join(dir_name, f)
            if os.path.isfile(fullpath):
                files.append(fullpath)
        return files
    else:
        return None

r"""
函数说明:
    功能：获取目录文件列表
    参数:
        dir_name：目录名
        exts:扩展名列表
    返回值:
        list or None
"""
def get_files_with_exts(dir_name,exts=['py']):
    if is_exists(dir_name):
        _files = get_files(dir_name)
        files_with_exts = [f for f in _files if exts is None or any(f.endswith(ext) for ext in exts)]
        return files_with_exts
    else:
        return None

r"""
函数说明:
    功能：获取目录文件基本名列表
    参数:
        dir_name：目录名
        exts:扩展名列表
    返回值:
        list or None
"""
def get_file_basename_with_exts(dir_name,exts=['py']):
    if is_exists(dir_name):
        _files = get_files_with_exts(dir_name,exts)
        return [os.path.splitext(get_basename(f))[0] for f in _files]
    else:
        return None

r"""
    函数说明:
        功能：正则查找文件名列表
        参数:
            dirpath：路径
            regex:正则表达式
        返回值:
            list or None
"""      
def get_files_with_regx(dirpath, regex):
    if is_exists(dirpath):
        fpaths =get_files(dirpath)
        match_objs, match_fpaths = [], []
        for i in range(len(fpaths)):
            match = re.search(regex, fpaths[i])
            if match is not None:
                match_objs.append(match)
                match_fpaths.append(fpaths[i])
        return match_objs, match_fpaths
    else:
        return None

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

r"""
    函数说明:
        功能：是否图像文件
        参数:
            filename：文件路径名
        返回值:
            bool
"""   
def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

r"""
    函数说明:
        功能：获取目录文件列表
        参数:
            dirpath：目录名
        返回值:
            bool
"""   
def get_images_from_dir(dirpath):
    images = []
    for f in os.listdir(dirpath):
        if is_image_file(f):
            images.append(os.path.join(dirpath, f))
    return images      


r"""
    函数说明:
        功能：读文本文件
        参数:
            path：文件全路径名
        返回值:
            bool
"""   
def read_text(path):
    return Path(path).open("r").read()


def read_lines(path):
    return Path(path).open("r").readlines()


def write_text(path, text, append=False):
    return Path(path).open("w+" if append else "w").write(text)


def write_lines(path, lines, append=False):
    with open(path, "w+" if append else "w") as file:
        file.writelines(lines)

def load_json(json_path):
    """
    Loads a json config from a file.
    """
    assert os.path.exists(json_path), "Json file %s not found" % json_path
    json_file = open(json_path)
    json_config = json_file.read()
    json_file.close()
    try:
        config = json.loads(json_config)
    except BaseException as err:
        raise Exception("Failed to validate config with error: %s" % str(err))

    return config


class PathHandler:
    """
    PathHandler is a base class that defines common I/O functionality for a URI
    protocol. It routes I/O for a generic URI which may look like "protocol://*"
    or a canonical filepath "/foo/bar/baz".
    """

    _strict_kwargs_check = True

    def _check_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Checks if the given arguments are empty. Throws a ValueError if strict
        kwargs checking is enabled and args are non-empty. If strict kwargs
        checking is disabled, only a warning is logged.

        Args:
            kwargs (Dict[str, Any])
        """
        if self._strict_kwargs_check:
            if len(kwargs) > 0:
                raise ValueError("Unused arguments: {}".format(kwargs))
        else:
            logger = logging.getLogger(__name__)
            for k, v in kwargs.items():
                logger.warning("[PathManager] {}={} argument ignored".format(k, v))

    def _get_supported_prefixes(self) -> List[str]:
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        raise NotImplementedError()

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, the cache stays on filesystem
        (under `file_io.get_cache_dir()`) and will be used by a different run.
        Therefore this function is meant to be used with read-only resources.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        raise NotImplementedError()

    def _copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> None:
        """
        Copies a local file to the specified URI.

        If the URI is another local path, this should be functionally identical
        to copy.

        Args:
            local_path (str): a file path which exists on the local file system
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing URI

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def _open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.

        Returns:
            file: a file-like object.
        """
        raise NotImplementedError()

    def _copy(
        self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def _exists(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        raise NotImplementedError()

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        raise NotImplementedError()

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        raise NotImplementedError()

    def _ls(self, path: str, **kwargs: Any) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        raise NotImplementedError()

    def _mkdirs(self, path: str, **kwargs: Any) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def _rm(self, path: str, **kwargs: Any) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def _symlink(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """
        Symlink the src_path to the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler to symlink from
            dst_path (str): A URI supported by this PathHandler to symlink to
        """
        raise NotImplementedError()


class NativePathHandler(PathHandler):
    """
    Handles paths that can be accessed using Python native system calls. This
    handler uses `open()` and `os.*` calls on the given path.
    """

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        self._check_kwargs(kwargs)
        return os.fspath(path)

    def _copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> None:
        self._check_kwargs(kwargs)
        assert self._copy(
            src_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
        )

    def _open(
        self,
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        closefd: bool = True,
        opener: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a path.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy works as follows:
                    * Binary files are buffered in fixed-size chunks; the size of
                    the buffer is chosen using a heuristic trying to determine the
                    underlying device’s “block size” and falling back on
                    io.DEFAULT_BUFFER_SIZE. On many systems, the buffer will
                    typically be 4096 or 8192 bytes long.
            encoding (Optional[str]): the name of the encoding used to decode or
                encode the file. This should only be used in text mode.
            errors (Optional[str]): an optional string that specifies how encoding
                and decoding errors are to be handled. This cannot be used in binary
                mode.
            newline (Optional[str]): controls how universal newlines mode works
                (it only applies to text mode). It can be None, '', '\n', '\r',
                and '\r\n'.
            closefd (bool): If closefd is False and a file descriptor rather than
                a filename was given, the underlying file descriptor will be kept
                open when the file is closed. If a filename is given closefd must
                be True (the default) otherwise an error will be raised.
            opener (Optional[Callable]): A custom opener can be used by passing
                a callable as opener. The underlying file descriptor for the file
                object is then obtained by calling opener with (file, flags).
                opener must return an open file descriptor (passing os.open as opener
                results in functionality similar to passing None).

            See https://docs.python.org/3/library/functions.html#open for details.

        Returns:
            file: a file-like object.
        """
        self._check_kwargs(kwargs)
        return open(  # type: ignore
            path,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        )

    def _copy(
        self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)

        if os.path.exists(dst_path) and not overwrite:
            logger = logging.getLogger(__name__)
            logger.error("Destination file {} already exists.".format(dst_path))
            return False

        try:
            shutil.copyfile(src_path, dst_path)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error("Error in file copy - {}".format(str(e)))
            return False

    def _symlink(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """
        Creates a symlink to the src_path at the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler

        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)
        logger = logging.getLogger(__name__)
        if not os.path.exists(src_path):
            logger.error("Source path {} does not exist".format(src_path))
            return False
        if os.path.exists(dst_path):
            logger.error("Destination path {} already exists.".format(dst_path))
            return False
        try:
            os.symlink(src_path, dst_path)
            return True
        except Exception as e:
            logger.error("Error in symlink - {}".format(str(e)))
            return False

    def _exists(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.exists(path)

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.isfile(path)

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.isdir(path)

    def _ls(self, path: str, **kwargs: Any) -> List[str]:
        self._check_kwargs(kwargs)
        return os.listdir(path)

    def _mkdirs(self, path: str, **kwargs: Any) -> None:
        self._check_kwargs(kwargs)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            # EEXIST it can still happen if multiple processes are creating the dir
            if e.errno != errno.EEXIST:
                raise

    def _rm(self, path: str, **kwargs: Any) -> None:
        self._check_kwargs(kwargs)
        os.remove(path)

class PathManager:
    """
    A class for users to open generic paths or translate generic paths to file names.
    """

    _PATH_HANDLERS: MutableMapping[str, PathHandler] = OrderedDict()
    _NATIVE_PATH_HANDLER = NativePathHandler()

    @staticmethod
    def __get_path_handler(path: Union[str, os.PathLike]) -> PathHandler:
        """
        Finds a PathHandler that supports the given path. Falls back to the native
        PathHandler if no other handler is found.

        Args:
            path (str or os.PathLike): URI path to resource

        Returns:
            handler (PathHandler)
        """
        path = os.fspath(path)  # pyre-ignore
        for p in PathManager._PATH_HANDLERS.keys():
            if path.startswith(p):
                return PathManager._PATH_HANDLERS[p]
        return PathManager._NATIVE_PATH_HANDLER

    @staticmethod
    def open(
        path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.

        Returns:
            file: a file-like object.
        """
        return PathManager.__get_path_handler(path)._open(  # type: ignore
            path, mode, buffering=buffering, **kwargs
        )

    @staticmethod
    def copy(
        src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """

        # Copying across handlers is not supported.
        assert PathManager.__get_path_handler(  # type: ignore
            src_path
        ) == PathManager.__get_path_handler(dst_path)
        return PathManager.__get_path_handler(src_path)._copy(
            src_path, dst_path, overwrite, **kwargs
        )

    @staticmethod
    def get_local_path(path: str, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        path = os.fspath(path)
        return PathManager.__get_path_handler(  # type: ignore
            path
        )._get_local_path(
            path, **kwargs
        )

    @staticmethod
    def copy_from_local(
        local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> None:
        """
        Copies a local file to the specified URI.

        If the URI is another local path, this should be functionally identical
        to copy.

        Args:
            local_path (str): a file path which exists on the local file system
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing URI

        Returns:
            status (bool): True on success
        """
        assert os.path.exists(local_path)
        return PathManager.__get_path_handler(dst_path)._copy_from_local(
            local_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
        )

    @staticmethod
    def exists(path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        return PathManager.__get_path_handler(path)._exists(  # type: ignore
            path, **kwargs
        )

    @staticmethod
    def isfile(path: str, **kwargs: Any) -> bool:
        """
        Checks if there the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        return PathManager.__get_path_handler(path)._isfile(  # type: ignore
            path, **kwargs
        )

    @staticmethod
    def isdir(path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        return PathManager.__get_path_handler(path)._isdir(  # type: ignore
            path, **kwargs
        )

    @staticmethod
    def ls(path: str, **kwargs: Any) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        return PathManager.__get_path_handler(path)._ls(path, **kwargs)

    @staticmethod
    def mkdirs(path: str, **kwargs: Any) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return PathManager.__get_path_handler(path)._mkdirs(  # type: ignore
            path, **kwargs
        )

    @staticmethod
    def rm(path: str, **kwargs: Any) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return PathManager.__get_path_handler(path)._rm(  # type: ignore
            path, **kwargs
        )

    @staticmethod
    def symlink(src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """Symlink the src_path to the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler to symlink from
            dst_path (str): A URI supported by this PathHandler to symlink to
        """
        # Copying across handlers is not supported.
        assert PathManager.__get_path_handler(  # type: ignore
            src_path
        ) == PathManager.__get_path_handler(dst_path)
        return PathManager.__get_path_handler(src_path)._symlink(
            src_path, dst_path, **kwargs
        )

    @staticmethod
    def register_handler(handler: PathHandler, allow_override: bool = False) -> None:
        """
        Register a path handler associated with `handler._get_supported_prefixes`
        URI prefixes.

        Args:
            handler (PathHandler)
            allow_override (bool): allow overriding existing handler for prefix
        """
        assert isinstance(handler, PathHandler), handler
        for prefix in handler._get_supported_prefixes():
            if not allow_override:
                assert prefix not in PathManager._PATH_HANDLERS
            PathManager._PATH_HANDLERS[prefix] = handler

        # Sort path handlers in reverse order so longer prefixes take priority,
        # eg: http://foo/bar before http://foo
        PathManager._PATH_HANDLERS = OrderedDict(
            sorted(PathManager._PATH_HANDLERS.items(), key=lambda t: t[0], reverse=True)
        )

    @staticmethod
    def set_strict_kwargs_checking(enable: bool) -> None:
        """
        Toggles strict kwargs checking. If enabled, a ValueError is thrown if any
        unused parameters are passed to a PathHandler function. If disabled, only
        a warning is given.

        With a centralized file API, there's a tradeoff of convenience and
        correctness delegating arguments to the proper I/O layers. An underlying
        `PathHandler` may support custom arguments which should not be statically
        exposed on the `PathManager` function. For example, a custom `HTTPURLHandler`
        may want to expose a `cache_timeout` argument for `open()` which specifies
        how old a locally cached resource can be before it's refetched from the
        remote server. This argument would not make sense for a `NativePathHandler`.
        If strict kwargs checking is disabled, `cache_timeout` can be passed to
        `PathManager.open` which will forward the arguments to the underlying
        handler. By default, checking is enabled since it is innately unsafe:
        multiple `PathHandler`s could reuse arguments with different semantic
        meanings or types.

        Args:
            enable (bool)
        """
        PathManager._NATIVE_PATH_HANDLER._strict_kwargs_check = enable
        for handler in PathManager._PATH_HANDLERS.values():
            handler._strict_kwargs_check = enable




