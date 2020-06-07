import jsonlines

class JsonlReader:
    """A class for reading jsonl files easily

    Attributes
    ----------
    _path : str
        The file location of the jsonl file
    _reader: jsonlines.Reader object
        The object which reads the files and keeps a tracker on the file
    _size : int
        Number of lines to be read
    _read : int
        Number of lines read so far

    Methods
    -------
    size()
        Getter method for _size.
    path()
        Getter method for _path.
    resetReader()
        Resets the _reader, starts over again.
    readNext(loop=False)
        Read the next line from the file. Loop over if loop=True.
    read(count=None, loop=False)
        Read the next count lines from the file, until self._size lines are read in total. Loop over if loop=True.
    close()
        Closes the reader of the file.
    """

    def __init__(self,path,size=None):
        """
        Initializes the JsonReader object.

        Parameters
        ----------
        path : string
            The path to the .jsonl file.
        size : int, optional
            The number of lines to be read from the file.
            If None, size is set to total number of lines.
            (default is None).
        """

        self._path = path
        self._reader = jsonlines.open(path,'r')
        if(size):
            self._size = size
        else:
            self._size = len(list(iter(jsonlines.open(path,'r'))))
        self._read = 0
    @property
    def size(self):
        """
        Function to get the _size attribute.

        Returns
        -------
            self._size : int
                The number of lines which will be read from the file
        """
        return self._size

    @property
    def path(self):
        """
        Function to get the _path attribute.
                Returns
        -------
            self._path : str
                The path of the the file
        """

        return self._path

    def resetReader(self):
        """Function to reset the reader attribute

        This function starts reading the file from the beginning.
        """

        self._reader = jsonlines.open(self._path,'r')
        self._read = 0
    def readNext(self,loop=False):
        """
        Function to read the next line from the path.

        If the loop parameter is set to True,
        then the reader starts over if the file ends.

        Raises an Exception if the file ends,
        and if the loop parameter is False.


        Parameters
        ----------
            loop : bool, optional
               Signifies whether the reader should start reading again
               if the end of file is reached. (default is False)
        Returns
        -------
            result: dict
                Returns next line of the file as a dict.
        """
        if(self._read>=self._size):
            print("Max count reached. Reset to start again.")
            return
        try:
            result = self._reader.read()
            self._read+=1
        except EOFError as e:
            if(loop==False):
                print('End of File reached.')
                return
            else:
                self.resetReader()
                try:
                    result =  self._reader.read()
                    self._read+=1
                except:
                    print('Empty File. Aborting Read')
                    return
        return result
    def read(self,count=None,loop=False):
        """
        Function to read the next count lines from the file.

        If the loop parameter is set to True,
        then the reader starts over if the file ends.

        Raises an Exception if the file ends,
        and returns the collected lines.


        Parameters
        ----------
            count : int, optional
                The number of lines to be read. If None,
                reads all the remaining lines. (default is None)

            loop : bool, optional
               Signifies whether the reader should start reading again
               if the end of file is reached. (default is False).
               Insignificant if the count is None.
        Returns
        -------
            lines: list of dict
                Returns the lines collected from the file.
        """

        lines = []
        if(count is None):
            for line in self._reader:
                self._read+=1
                if(self._read<=self._size):
                    lines.append(line)
                else:
                    break
            return lines
        for line in range(count):
            try:
                result = self.readNext(loop=loop)
                if(result):
                    lines.append(result)
                else:
                    raise Exception('None received at self.readNext')
            except:
                print('Error occurred. Returning collected lines so far.')
        return lines
    def close(self):
        """ Function to close the reader """
        self._reader.close()
