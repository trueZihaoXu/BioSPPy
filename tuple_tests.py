from biosppy.utils import ReturnTuple
import numpy as np


T = ReturnTuple((1, 2), ('a', 'b'))
dictionary = {'cc': 3, 'dd': 4}
array = np.array([1, 2, 3])

#%% POSITIVE TESTS
# 1 - append single value
T.append(3, 'cc')
T.append([3], ['c'])

#%% 2 - append multiple values with list
T.append([3, 4], ['cc', 'dd'])

#%% 3 - append multiple values with tuple
T.append((3, 4), ('cc', 'dd'))

#%% 4 - append multiple values with dict
T.append(dictionary)

#%% 5 - append array
T.append(array, 'array')

#%% 6- append dictionary
T.append(dictionary, 'dict')

#%% 7- append dictionary and array
T.append([dictionary, array, 2.14, 'hey', True], ['dict', 'array', 'hey', 'hey2', 'he3'])

#%% NEGATIVE TESTS
# 8 - user gives only a single parameter
T.append(3)

#%% 9 - key missing
T.append([3, 4], ['a'])

#%% 10 - append ReturnTuple
T.append(T, 't')




