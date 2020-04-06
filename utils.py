import os, json
import numpy as np

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
