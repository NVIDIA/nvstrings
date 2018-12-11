import pyniNVStrings


def to_device(strs):
    """Create nvstrings instance from list of Python strings."""
    cptr = pyniNVStrings.n_createFromHostStrings(strs)
    return nvstrings(cptr)


def from_csv(csv, column, lines=0):
    """
    This is not full featured and will be removed in the future.

    Parameters
    ----------
        csv : str
            Path to the csv file from which to load data
        column : int
            0-based index of the column to read into an nvstrings object
        lines : int
            maximum number of lines to read from the file
    Returns
    -------
    A new nvstrings instance pointing to strings loaded onto the GPU

    Examples
    --------

    .. code-block:: python

      import nvstrings
      strs = nvstrings.from_csv("file.csv",2)

    """
    rtn = pyniNVStrings.n_createFromCSV(csv, column, lines)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def free(dstrs):
    """Force free resources for the specified instance."""
    pyniNVStrings.n_destroyStrings(dstrs.m_cptr)
    dstrs.m_cptr = 0


def bind_cpointer(cptr):
    """Bind an NVStrings C-pointer to a new instance."""
    rtn = None
    if cptr is not 0:
        rtn = nvstrings(cptr)
    return rtn


# this will be documented with all the public methods
class nvstrings:
    """
    Instance manages a list of strings in device memory.

    Operations are across all of the strings and their results reside in device
    memory.

    Strings in the list are immutable.

    Methods that modify any string will create new nvstrings instance.
    """
    #
    m_cptr = 0

    def __init__(self, cptr):
        """
        Use to_device() to create new instance from Python array of strings.
        """
        self.m_cptr = cptr

    def __del__(self):
        pyniNVStrings.n_destroyStrings(self.m_cptr)
        self.m_cptr = 0

    def __str__(self):
        return str(pyniNVStrings.n_createHostStrings(self.m_cptr))

    def __repr__(self):
        return "<nvstrings count={}>".format(self.size())

    def to_host(self):
        """
        Copies strings back to CPU memory into a Python array.

        Returns
        -------
        A list of strings

        Examples
        --------

        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])

          h = s.upper().to_host()
          print(h)

        Output:

        .. code-block:: python

          ["HELLO","WORLD"]

        """
        return pyniNVStrings.n_createHostStrings(self.m_cptr)

    def size(self):
        """
        The number of strings managed by this instance.

        Returns
        -------
          int: number of strings

        Examples
        --------

        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])
          len(s)

        Output:

        .. code-block:: python

          2

        """
        return pyniNVStrings.n_size(self.m_cptr)

    def len(self, devptr=0):
        """
        Returns the lengths in characters of each string.

        Parameters
        ----------
            devptr : GPU memory pointer
                Pointer to GPU memory where length values should be written

        Examples
        --------

        .. code-block:: python

          import nvstrings
          import numpy as np
          from librmm_cffi import librmm

          # example passing device memory pointer
          s = nvstrings.to_device(["abc","d","ef"])
          arr = np.arange(s.size(),dtype=np.int32)
          d_arr = librmm.to_device(arr)
          s.len(d_arr.device_ctypes_pointer.value)
          print(d_arr.copy_to_host())

        Output:

        .. code-block:: python

          [3,1,2]

        """
        rtn = pyniNVStrings.n_len(self.m_cptr, devptr)
        return rtn

    # def get_nulls( self, en=False, devptr=0 ):
    #     """Returns the number of null strings in this instance."""
    #     rtn = pyniNVStrings.n_get_nulls(self.m_cptr,en,devptr)
    #     return rtn

    def compare(self, str, devptr=0):
        """
        Compare each string to the supplied string.

        Returns value of 0 where string matches.

        Returns < 0 when first different character is lower than argument
        string or argument string is shorter.

        Returns > 0 when first different character is greater than the argument
        string or the argument string is longer.

        Parameters
        ----------
            str : str
                String to compare all strings in an nvstrings object to

            devptr : GPU memory pointer
                Pointer to GPU array where length values should be written

        Examples
        --------

        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])

          print(s.compare('hello'))

        Output:

        .. code-block:: python

          [0,15]

        """
        rtn = pyniNVStrings.n_compare(self.m_cptr, str, devptr)
        return rtn

    def hash(self, devptr=0):
        """
        Returns hash values represented by each string.

        Parameters
        ----------
            devptr : GPU memory pointer
                Pointer to GPU array where length values should be written

        Examples
        --------

        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])
          s.hash()

        Output:

        .. code-block:: python

          [99162322, 113318802]

        """
        rtn = pyniNVStrings.n_hash(self.m_cptr, devptr)
        return rtn

    def stoi(self, devptr=0):
        """
        Returns integer value represented by each string.

        Parameters
        ----------
            devptr : GPU memory pointer
                Pointer to GPU array where length values should be written

        Examples
        --------
        .. code-block:: python

          import nvstrings
          import numpy as np
          s = nvstrings.to_device([
            "1234","5678","90",None,"-876","543.2","-0.12",".55","-.002",""
            ])
          s.stoi()

        Output:

        .. code-block:: python

          [1234, 5678, 90, 0, -876, 543, 0, 0, 0, 0]

        """
        rtn = pyniNVStrings.n_stoi(self.m_cptr, devptr)
        return rtn

    def stof(self, devptr=0):
        """
        Returns float values represented by each string.

        Parameters
        ----------
            devptr : GPU memory pointer
                Pointer to GPU array where length values should be written

        Examples
        --------
        .. code-block:: python

          import nvstrings
          import numpy as np
          from librmm_cffi import librmm

          s = nvstrings.to_device([
            "123", None, "-876", "43.2", "-0.12", ".55", "-.002", ""
          ])
          for s in s.stof(): print(s)

        Output:

        .. code-block:: python

          123.0
          0.0
          -876.0
          43.20000076293945
          -0.11999999731779099
          0.550000011920929
          -0.001999999862164259
          0.0

        """
        rtn = pyniNVStrings.n_stof(self.m_cptr, devptr)
        return rtn

    def cat(self, others=None, sep=None, na_rep=None):
        """
        Appends the given strings to this list of strings and returns a new
        nvstrings.

        Parameters
        ----------
            others : List of str
                Strings to be appended. The number of strings must match.

                This must be either a Python array of strings or another
                nvstrings instance.

            sep : str
                If specified, this separator will be appended to each string
                before appending the others.

            na_rep : char
                This character will take the place of any null strings (not
                empty strings) in either list.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s1 = nvstrings.to_device(['hello', None,'goodbye'])
          s2 = nvstrings.to_device(['world','globe', None])

          print(s1.cat(s2,sep=':', na_rep='_'))

        Output:

        .. code-block:: python

          ["hello:world","_:globe","goodbye:_"]

        """
        rtn = pyniNVStrings.n_cat(self.m_cptr, others, sep, na_rep)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def join(self, sep=''):
        """
        Concatentate a list of strings into a single string.

        Parameters
        ----------
            sep : str
                this separator will be appended to each string before appending
                the others.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello",None,"goodbye"])
          s.join(sep=':')

        Output:

        .. code-block:: python

          ['hello:goodbye']

        """
        rtn = pyniNVStrings.n_join(self.m_cptr, sep)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def split(self, delimiter=None, n=-1):
        """
        Returns an array of nvstrings each representing the split of each
        individual string.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of each string.
                Default is space.

            n : int
                Maximum number of strings to return for each split.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello world","goodbye","well said"])
          for result in s.split(' '):
            print(result)

        Output:

        .. code-block:: python

          ["hello","world"]
          ["goodbye"]
          ["well","said"]

        """
        strs = pyniNVStrings.n_split(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def rsplit(self, delimiter=None, n=-1):
        """
        Returns an array of nvstrings each representing the split of each
        individual string. Delimiter is searched for from the end.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of each string.
                Default is space.

            n : int
                Maximum number of strings to return for each split.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello world","goodbye","well said"])
          for result in s.rsplit(' '):
            print(result)

        Output:

        .. code-block:: python

          ["hello","world"]
          ["goodbye"]
          ["well","said"]

        """
        strs = pyniNVStrings.n_rsplit(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def partition(self, delimiter=' '):
        """
        Each string is split into two strings on the first delimiter found.

        Three strings are returned for each string: beginning, delimiter, end.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of each string.
                Default is space.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello world","goodbye","well said"])
          for result in s.partition(','):
            print(result)

        Output:

        .. code-block:: python

          ["hello","world"]
          ["goodbye"]
          ["well","said"]

        """
        strs = pyniNVStrings.n_partition(self.m_cptr, delimiter)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def rpartition(self, delimiter=' '):
        """
        Each string is split into two strings on the first delimiter found.
        Delimiter is searched for from the end.

        Three strings are returned for each string: beginning, delimiter, end.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of each string.
                Default is space.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello world","goodbye","well said"])
          for result in s.rpartition(','):
            print(result)

        Output:

        .. code-block:: python

          ["hello","world"]
          ["goodbye"]
          ["well","said"]

        """
        strs = pyniNVStrings.n_rpartition(self.m_cptr, delimiter)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def split_column(self, delimiter=' ', n=-1):
        """
        A new set of columns (nvstrings) is created by splitting the strings
        vertically.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of each string.
                Default is space.

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello world","goodbye","well said"])
          for result in s.split_column(' '):
            print(result)

        Output:

        .. code-block:: python

          ["hello","goodbye","well"]
          ["world",None,"said"]

        """
        strs = pyniNVStrings.n_split_column(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def rsplit_column(self, delimiter=' ', n=-1):
        """
        A new set of columns (nvstrings) is created by splitting the strings
        vertically. Delimiter is searched from the end.

        Parameters
        ----------
            delimiter : str
                The character used to locate the split points of each string.
                Default is space.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello world","goodbye","well said"])
          for result in s.rsplit_column(' '):
            print(result)

        Output:

        .. code-block:: python

          ["hello","goodbye","well"]
          ["world",None,"said"]

        """
        strs = pyniNVStrings.n_rsplit_column(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def get(self, i):
        """
        Returns the character specified in each string as a new string.

        The nvstrings returned contains a list of single character strings.

        Parameters
        ----------
          i : int
            The character position identifying the character in each string to
            return.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello world","goodbye","well said"])
          print(s.get(0))

        Output:

        .. code-block:: python

          ['h', 'g', 'w']

        """
        rtn = pyniNVStrings.n_get(self.m_cptr, i)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def repeat(self, repeats):
        """
        Appends each string with itself the specified number of times. This
        returns a nvstrings instance with the new strings.

        Parameters
        ----------
            repeats : int
               The number of times each string should be repeated. Repeat count
               of 0 or 1 will just return copy of each string.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye","well"])
          print(s.repeat(2))

        Output:

        .. code-block:: python

          ['hellohello', 'goodbyegoodbye', 'wellwell']

        """
        rtn = pyniNVStrings.n_repeat(self.m_cptr, repeats)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def pad(self, width, side='left', fillchar=' '):
        """
        Add specified padding to each string. Side:{'left','right','both'},
        default is 'left'.

        Parameters
        ----------
          fillchar : char
            The character used to do the padding. Default is space character.
            Only the first character is used.

          side : str
            Either one of "left", "right", "both". The default is "left"

            "left" performs a padding on the left – same as rjust()

            "right" performs a padding on the right – same as ljust()

            "both" performs equal padding on left and right – sames as center()

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye","well"])
          print(s.pad(' ', side='left'))

        Output:

        .. code-block:: python

          [" hello"," goodbye"," well"]

        """
        rtn = pyniNVStrings.n_pad(self.m_cptr, width, side, fillchar)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def ljust(self, width, fillchar=' '):
        """
        Pad the end of each string to the minimum width.

        Parameters
        ----------
          width : int
            The minimum width of characters of the new string. If the width is
            smaller than the existing string, no padding is performed.

          fillchar : char
            The character used to do the padding. Default is space character.
            Only the first character is used.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye","well"])
          print(s.ljust(width=6))

        Output:

        .. code-block:: python

          ['hello ', 'goodbye', 'well  ']

        """
        rtn = pyniNVStrings.n_ljust(self.m_cptr, width)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def center(self, width, fillchar=' '):
        """
        Pad the beginning and end of each string to the minimum width.

        Parameters
        ----------
          width : int
            The minimum width of characters of the new string. If the width is
            smaller than the existing string, no padding is performed.

          fillchar : char
            The character used to do the padding. Default is space character.
            Only the first character is used.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye","well"])
          for result in s.center(width=6):
            print(result)

        Output:

        .. code-block:: python

          ['hello ', 'goodbye', ' well ']

        """
        rtn = pyniNVStrings.n_center(self.m_cptr, width, fillchar)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def rjust(self, width, fillchar=' '):
        """
        Pad the beginning of each string to the minimum width.

        Parameters
        ----------
          width : int
            The minimum width of characters of the new string. If the width is
            smaller than the existing string, no padding is performed.

          fillchar : char
            The character used to do the padding. Default is space character.
            Only the first character is used.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye","well"])
          print(s.ljust(width=6))

        Output:

        .. code-block:: python

          [' hello', 'goodbye', '  well']

        """
        rtn = pyniNVStrings.n_rjust(self.m_cptr, width)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def zfill(self, width):
        """
        Pads the strings with leading zeros. It will handle prefix sign
        characters correctly for strings that are numbers.

        Parameters
        ----------
          width : int
            The minimum width of characters of the new string. If the width is
            smaller than the existing string, no padding is performed.
        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye","well"])
          print(s.zfill(width=6))

        Output:

        .. code-block:: python

          ['0hello', 'goodbye', '00well']

        """
        rtn = pyniNVStrings.n_zfill(self.m_cptr, width)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def wrap(self, width):
        """
        This will place new-line characters in whitespace so each line is no
        more than width characters. Lines will not be truncated.

        Parameters
        ----------
          width : int
            The maximum width of characters per newline in the new string. If
            the width is smaller than the existing string, no newlines will be
            inserted.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello there","goodbye all","well ok"])
          print(s.wrap(3))

        Output:

        .. code-block:: python

          ['hello\\nthere', 'goodbye\\nall', 'well\\nok']

        """
        rtn = pyniNVStrings.n_wrap(self.m_cptr, width)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def slice(self, start, stop=None, step=None):
        """
        Returns a substring of each string.

        Parameters
        ----------
        start : int
          Beginning position of the string to extract. Default is beginning of
          each string.
        stop : int
          Ending position of the string to extract. Default is end of each
          string.
        step : str
          Characters that are to be captured within the specified section.
          Default is every character.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye"])
          print(s.slice(2,5))

        Output:

        .. code-block:: python

          ['llo', 'odb']

        """
        rtn = pyniNVStrings.n_slice(self.m_cptr, start, stop, step)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def slice_from(self, starts=0, stops=0):
        """
        Return substring of each string using positions for each string.
        Position values must be in device memory.

        """
        rtn = pyniNVStrings.n_slice_from(self.m_cptr, starts, stops)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def slice_replace(self, start=None, stop=None, repl=None):
        """
        Replace the specified section of each string with a new string.
        """
        rtn = pyniNVStrings.n_slice_replace(self.m_cptr, start, stop, repl)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def replace(self, pat, repl, n=-1, regex=True):
        """
        Replace a string (pat) found in each string with another string (repl).

        Parameters
        ----------
        pat : str
          String to be replaced.
          This can also be a regex expression -- not a compiled regex.

        repl : str
          String to replace `strng` with

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye"])
          print(s.replace('e', ''))

        Output:

        .. code-block:: python

          ['hllo', 'goodby']

        """
        rtn = pyniNVStrings.n_replace(self.m_cptr, pat, repl, n, regex)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def lstrip(self, to_strip=None):
        """
        Strip leading characters from each string.

        Parameters
        ----------
        to_strip : str
          Characters to be removed from leading edge of each string

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye"])
          print(s.lstrip('h', ''))

        Output:

        .. code-block:: python

          ['ello', 'goodbye']

        """
        rtn = pyniNVStrings.n_lstrip(self.m_cptr, to_strip)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def strip(self, to_strip=None):
        """
        Strip leading and trailing characters from each string.

        Parameters
        ----------
        to_strip : str
          Characters to be removed from both edges of each string

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["oh, hello","goodbye"])
          print(s.strip('o', ''))

        Output:

        .. code-block:: python

          ['h, hell', 'goodbye']

        """
        rtn = pyniNVStrings.n_strip(self.m_cptr, to_strip)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def rstrip(self, to_strip=None):
        """
        Strip trailing characters from each string.

        Parameters
        ----------
        to_strip : str
          Characters to be removed from trailing edge of each string

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","goodbye"])
          print(s.rstrip('o', ''))

        Output:

        .. code-block:: python

          ['hell', 'goodbye']

        """
        rtn = pyniNVStrings.n_rstrip(self.m_cptr, to_strip)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def lower(self):
        """
        Convert each string to lowercase.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["Hello, Friend","Goodbye, Friend"])
          print(s.lower())

        Output:

        .. code-block:: python

          ['hello, friend', 'goodbye, friend']

        """
        rtn = pyniNVStrings.n_lower(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def upper(self):
        """
        Convert each string to uppercase.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello, friend","goodbye, friend"])
          print(s.lower())

        Output:

        .. code-block:: python

          ['HELLO, FRIEND', 'GOODBYE, FRIEND']

        """
        rtn = pyniNVStrings.n_upper(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def capitalize(self):
        """
        Capitalize first character of each string.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello, friend","goodbye, friend"])
          print(s.lower())

        Output:

        .. code-block:: python

          ['Hello, friend", "Goodbye, friend"]

        """
        rtn = pyniNVStrings.n_capitalize(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def swapcase(self):
        """
        Change each lowercase character to uppercase and vice versa.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["Hello, Friend","Goodbye, Friend"])
          print(s.lower())

        Output:

        .. code-block:: python

          ['hELLO, fRIEND', 'gOODBYE, fRIEND']

        """
        rtn = pyniNVStrings.n_swapcase(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def title(self):
        """
        Uppercase the first letter of each word and lowercase the rest.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello, Friend","goodbye, Friend"])
          print(s.lower())

        Output:

        .. code-block:: python

          ['Hello, friend', 'Goodbye, friend']

        """
        rtn = pyniNVStrings.n_title(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def index(self, search, start=0, end=None, devptr=0):
        """
        Same as find but throws an error if arg is not found in all strings.

        Parameters
        ----------
          search : str
            String to find

          start : int
            Beginning of section to replace. Default is beginning of each
            string.

          end : int
            End of section to replace. Default is end of each string.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.

            Memory size must be able to hold at least size() of int32 values.

            If `devptr` not given, results returned as list of ints.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])

          print(s.index('l'))

        Output:

        .. code-block:: python

          [2,3]

        """
        rtn = pyniNVStrings.n_index(self.m_cptr, search, start, end, devptr)
        return rtn

    def rindex(self, search, start=0, end=None, devptr=0):
        """
        Same as rfind but throws an error if arg is not found in all strings.

        Parameters
        ----------
          search : str
            String to find

          start : int
            Beginning of section to replace. Default is start of each string

          end : int
            End of section to replace. Default is end of each string.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.

            Memory size must be able to hold at least size() of int32 values.

            If `devptr` not given, results returned as list of ints.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","world"])

          print(s.rindex('l'))

        Output:

        .. code-block:: python

          [3,3]

        """
        rtn = pyniNVStrings.n_rindex(self.m_cptr, search, start, end, devptr)
        return rtn

    def find(self, search, start=0, end=None, devptr=0):
        """
        Find the specified string within each string. Return -1 for those
        strings where the arg is not found.

        Parameters
        ----------
          search : str
            String to find

          start : int
            Beginning of section to replace. Default is start of each string

          end : int
            End of section to replace. Default is end of each string.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.

            Memory size must be able to hold at least size() of int32 values.

            If `devptr` not given, results returned as list
            of integers.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.find('o'))

        Output:

        .. code-block:: python

          [4,-1,1]

        """
        rtn = pyniNVStrings.n_find(self.m_cptr, search, start, end, devptr)
        return rtn

    def find_from(self, search, starts=0, ends=0, devptr=0):
        """
        Find the specified string within each string starting at individual
        character positions.

        The starts and ends parameters must be device memory pointers.

        Return -1 for those strings where the arg is not found.

        Parameters
        ----------
          search : str
            String to find

          starts : GPU memory pointer
            Pointer to GPU array of ints of beginning of sections to replace,
            one per string. Default is beginning of each string.

          ends : GPU memory pointer
            Pointer to GPU array of ints of end of sections to replace.
            Default is end of each string.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.

            Memory size must be able to hold at least size() of int32 values.

            If `devptr` not given, results returned as list of ints.

        """
        rtn = pyniNVStrings.n_find_from(self.m_cptr, search,
                                        starts, ends, devptr)
        return rtn

    def rfind(self, search, start=0, end=None, devptr=0):
        """
        Find the specified string within each string. Search from the end
        of the string.

        Return -1 for those strings where the arg is not found.

        Parameters
        ----------
          search : str
            String to find

          start : int
            Beginning of section to replace. Default is beginning of each
            string.

          end : int
            End of section to replace. Default is end of each string.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.

            Memory size must be able to hold at least size() of int32 values.

            If `devptr` not given, results returned as list of ints.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.rfind('o'))

        Output:

        .. code-block:: python

          [4, -1, 1]

        """
        rtn = pyniNVStrings.n_rfind(self.m_cptr, search, start, end, devptr)
        return rtn

    def findall(self, pat):
        """
        A new array of nvstrings is created for each string in this instance.

        Parameters
        ----------
            pat : str
                The regex pattern used to search for substrings

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hare","bunny","rabbit"])
          for result in s.findall('[ab]'):
            print(result)

        Output:

        .. code-block:: python

          ["a"]
          ["b"]
          ["a","b","b"]

        """
        strs = pyniNVStrings.n_findall(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def findall_column(self, pat):
        """
        A new set of nvstrings is created by organizing substring
        results vertically.

        Parameters
        ----------
            pat : str
                The regex pattern to search for substrings

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hare","bunny","rabbit"])
          for result in s.findall_column('[ab]'):
            print(result)

        Output:

        .. code-block:: python

          ["a","b","a"]
          [None,None,"b"]
          [None,None,"b"]

        """
        strs = pyniNVStrings.n_findall_column(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def contains(self, pat, regex=True, devptr=0):
        """
        Find the specified string within each string.

        Parameters
        ----------
          pat : str
            Pattern to find

          regex : bool
            If `True`, pat is interpreted as a regex
            If `False`, pat is a substring to to be searched for

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.

            Memory size must be able to hold at least size() of int32 values.

            If `devptr` not given, results are returned as a list of ints.

        Returns
        -------
        `True` if `pat` found, `False` if not

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.contains('o', regex=False))

        Output:

        .. code-block:: python

          [True, False, True]

        """
        rtn = pyniNVStrings.n_contains(self.m_cptr, pat, regex, devptr)
        return rtn

    def match(self, pat, devptr=0):
        """
        The specified pattern must match the beginning of each string.

        Parameters
        ----------
          pat : str
            Pattern to find

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.

            Memory size must be able to hold at least size() of int32 values.

            If `devptr` not given, results are returned as list of ints.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.match('h'))

        Output:

        .. code-block:: python

          [True, False, True]

        """
        rtn = pyniNVStrings.n_match(self.m_cptr, pat, devptr)
        return rtn

    def count(self, pat, devptr=0):
        """
        Count occurrences of pattern in each string.

        Parameters
        ----------
          pat : str
            Pattern to find

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.

            Memory size must be able to hold at least size() of int32 values.

            If `devptr` not given, results are returned as list of ints.
        """
        rtn = pyniNVStrings.n_count(self.m_cptr, pat, devptr)
        return rtn

    def startswith(self, pat, devptr=0):
        """
        Return true for the strings where the specified string is at the
        beginning.

        Parameters
        ----------
          pat : str
            Pattern to find. Regular expressions are not accepted.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.

            Memory size must be able to hold at least size() of int32 values.

            If `devptr` not given, results returned as list of ints.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.startswith('h'))

        Output:

        .. code-block:: python

          [True, False, False]

        """
        rtn = pyniNVStrings.n_startswith(self.m_cptr, pat, devptr)
        return rtn

    def endswith(self, pat, devptr=0):
        """
        Return true for the strings where the specified string is at the end.

        Parameters
        ----------
          pat : str
            Pattern to find. Regular expressions are not accepted.

          devptr : GPU memory pointer
            Optional device memory pointer to hold the results.

            Memory size must be able to hold at least size() of int32 values.

            If pointer is not provided, the results are returned as Python
            array of integers.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.endsswith('d'))

        Output:

        .. code-block:: python

          [False, False, True]

        """
        rtn = pyniNVStrings.n_endswith(self.m_cptr, pat, devptr)
        return rtn

    def extract(self, pat):
        """
        A new array of nvstrings is created for each string in this instance.

        Parameters
        ----------
            pat : str
                The regex pattern with group capture syntax

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["a1","b2","c3"])
          for result in s.extract('([ab])(\d)'):
            print(result)

        Output:

        .. code-block:: python

          ["a","1"]
          ["b","2"]
          [None,None]

        """
        strs = pyniNVStrings.n_extract(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def extract_column(self, pat):
        """
        A new vector of nvstrings is created by organizing group results
        vertically.

        Parameters
        ----------
            pat : str
                The regex pattern with group capture syntax

        Examples
        --------

        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["a1", "b2", "c3"])
          for result in s.extract_column('([ab])(\d)'):
            print(result)

        Output:

        .. code-block:: python

          ["a","b"]
          ["1","2"]
          [None,None]

        """
        strs = pyniNVStrings.n_extract_column(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def isalnum(self, devptr=0):
        """
        Return true for strings that contain only alpha-numeric characters.
        Equivalent to: isalpha() or isdigit() or isnumeric() or isdecimal()
        """
        rtn = pyniNVStrings.n_isalnum(self.m_cptr, devptr)
        return rtn

    def isalpha(self, devptr=0):
        """
        Return true for strings that contain only alphabetic characters.
        """
        rtn = pyniNVStrings.n_isalpha(self.m_cptr, devptr)
        return rtn

    def isdigit(self, devptr=0):
        """
        Return true for strings that contain only decimal and digit characters.
        """
        rtn = pyniNVStrings.n_isdigit(self.m_cptr, devptr)
        return rtn

    def isspace(self, devptr=0):
        """
        Return true for strings that contain only whitespace characters.
        """
        rtn = pyniNVStrings.n_isspace(self.m_cptr, devptr)
        return rtn

    def isdecimal(self, devptr=0):
        """
        Return true for strings that contain only decimal characters -- those
        that can be used to extract base10 numbers.
        """
        rtn = pyniNVStrings.n_isdecimal(self.m_cptr, devptr)
        return rtn

    def isnumeric(self, devptr=0):
        """
        Return true for strings that contain only numeric characters.
        These include digit and numeric characters.
        """
        rtn = pyniNVStrings.n_isnumeric(self.m_cptr, devptr)
        return rtn

    def translate(self, table):
        """
        Translate individual characters to new characters using the provided
        table.

        Parameters
        ----------
          pat : dict
            Use str.maketrans() to build the mapping table. Unspecified
            characters are unchanged.

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["hello","there","world"])
          print(s.translate(str.maketrans('elh','ELH')))

        Output:

        .. code-block:: python

          ['HELLo', 'tHErE', 'worLd]

        """
        rtn = pyniNVStrings.n_translate(self.m_cptr, table)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def sort(self, stype, asc=True):
        """
        Sort this list by name (2) or length (1) or both (3).

        Parameters
        ----------
          stype : int
            Type of sort to use.

            If stype is 1, strings will be sorted by length

            If stype is 2, strings will be sorted alphabetically by name

            If stype is 3, strings will be sorted by length and then
            alphabetically

          asc : bool
            Whether to sort ascending (True) or descending (False)

        Examples
        --------
        .. code-block:: python

          import nvstrings

          s = nvstrings.to_device(["aaa", "bb", "aaaabb"])
          print(s.sort(3))

        Output:

        .. code-block:: python

          ['bb', 'aaa', 'aaaabb']

        """
        pyniNVStrings.n_sort(self.m_cptr, stype, asc)
        return self

    def sublist(self, indexes):
        """
        Return a sublist of strings from this instance.

        Parameters
        ----------
          indexes : List of ints
            0-based indexes of strings to return from an nvstrings object

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.sublist([0, 2]))

        Output:

        .. code-block:: python

          ['hello', 'world']

        """
        rtn = pyniNVStrings.n_sublist(self.m_cptr, indexes)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def remove_strings(self, indexes, count=0):
        """
        Remove the specified strings and return a new instance.

        Parameters
        ----------
          indexes : List of ints
            0-based indexes of strings to remove from an nvstrings object
            If `indexes` is a dev pointer, `count` argument is required.

          count : If a dev pointer, `count` is number of ints, else it is
          ignored.

        Examples
        --------
        .. code-block:: python

          import nvstrings
          s = nvstrings.to_device(["hello","there","world"])

          print(s.remove_strings([0, 2]))

        Output:

        .. code-block:: python

          ['there']

        """
        rtn = pyniNVStrings.n_remove_strings(self.m_cptr, indexes, count)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn
