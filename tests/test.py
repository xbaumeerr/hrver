from typing import overload
from multipledispatch import dispatch


@overload
@dispatch(int, int)
def add(x:int, y:int):
    return x + y

@dispatch(str, str)
def add(x:str, y:str):
    return x.upper() + y.upper()