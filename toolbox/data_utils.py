import os, json, validators, requests, warnings
import numpy as np

# for cache function
from functools import lru_cache, wraps
from datetime import datetime, timedelta

def timed_lru_cache(seconds: int, maxsize: int = 128):
    ''' add an expiry to the lru_cache
    ref: https://realpython.com/lru-cache-python/
    '''
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime
            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache

def JsonReader(fname, raise_error = False):
	'''
	Returns the JSON object stored inside the given fname
	'''
	if os.path.isfile(fname):
		with open(fname, 'r') as fh:
			return json.load(fh)
	elif validators.url(fname):
		r = requests.get(fname)
		return r.json()
	else:
		if raise_error:
			raise FileNotFoundError(f'JsonReader: {fname} is not found.')
		else:
			warnings.warn(f'JsonReader: {fname} is not found.')
			return None

def JsonLookUp(jsonObj, searchKey, searchVal, resultKey= None):
	'''
	Search the jsonObj for where the searchKey is the searchVal and return the
	resultKey value. If there's more than one object match, the function will
	return a list of dictionary. If there's more than one object match and resultKey
	is provided, than it will only return the value from the first object.

	Args:
		jsonObj: a simple dictionary or list of dictionary (each with the same keys)
		resultKey: if not given, the matching jsonObj will be returned
	'''
	outObj = None
	outVal = None

	if type(jsonObj) == list:
		outObj = [obj for obj in jsonObj if obj[searchKey] == searchVal]

		if len(outObj) == 0:
			return None
		else:
			outObj = outObj[0] if (len(outObj) == 1 or resultKey) else outObj

		if resultKey:
			outVal = outObj[resultKey]
	else:
		outObj = jsonObj if jsonObj[searchKey] == searchVal else None

		if resultKey and outObj:
			outVal = outObj[resultKey]

	if resultKey:
		return outVal
	else:
		return outObj
