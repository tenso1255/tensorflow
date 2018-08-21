/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/python/util/util.h"

#include <functional>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace tensorflow {
namespace swig {

namespace {

// Type object for collections.Sequence. This is set by RegisterSequenceClass.
PyObject* CollectionsSequenceType = nullptr;
// Type object for collections.Mapping, set by RegisterMappingClass.
PyObject* CollectionsMappingType = nullptr;
PyTypeObject* SparseTensorValueType = nullptr;

const int kMaxItemsInCache = 1024;

bool WarnedThatSetIsNotSequence = false;

bool IsString(PyObject* o) {
  return PyBytes_Check(o) ||
#if PY_MAJOR_VERSION < 3
         PyString_Check(o) ||
#endif
         PyUnicode_Check(o);
}

// Work around a writable-strings warning with Python 2's PyMapping_Keys macro,
// and while we're at it give them consistent behavior by making sure the
// returned value is a list.
//
// As with PyMapping_Keys, returns a new reference.
//
// On failure, returns nullptr.
PyObject* MappingKeys(PyObject* o) {
#if PY_MAJOR_VERSION >= 3
  return PyMapping_Keys(o);
#else
  static char key_method_name[] = "keys";
  Safe_PyObjectPtr raw_result(PyObject_CallMethod(o, key_method_name, nullptr));
  if (PyErr_Occurred() || raw_result.get() == nullptr) {
    return nullptr;
  }
  return PySequence_Fast(
      raw_result.get(),
      "The '.keys()' method of a custom mapping returned a non-sequence.");
#endif
}

// Equivalent to Python's 'o.__class__.__name__'
// Note that '__class__' attribute is set only in new-style classes.
// A lot of tensorflow code uses __class__ without checks, so it seems like
// we only support new-style classes.
StringPiece GetClassName(PyObject* o) {
  // __class__ is equivalent to type() for new style classes.
  // type() is equivalent to PyObject_Type()
  // (https://docs.python.org/3.5/c-api/object.html#c.PyObject_Type)
  // PyObject_Type() is equivalent to o->ob_type except for Py_INCREF, which
  // we don't need here.
  PyTypeObject* type = o->ob_type;

  // __name__ is the value of `tp_name` after the last '.'
  // (https://docs.python.org/2/c-api/typeobj.html#c.PyTypeObject.tp_name)
  StringPiece name(type->tp_name);
  size_t pos = name.rfind('.');
  if (pos != StringPiece::npos) {
    name.remove_prefix(pos + 1);
  }
  return name;
}

string PyObjectToString(PyObject* o) {
  if (o == nullptr) {
    return "<null object>";
  }
  PyObject* str = PyObject_Str(o);
  if (str) {
#if PY_MAJOR_VERSION < 3
    string s(PyString_AS_STRING(str));
#else
    string s(PyUnicode_AsUTF8(str));
#endif
    Py_DECREF(str);
    return tensorflow::strings::StrCat("type=", GetClassName(o), " str=", s);
  } else {
    return "<failed to execute str() on object>";
  }
}

class CachedTypeCheck {
 public:
  explicit CachedTypeCheck(std::function<int(PyObject*)> ternary_predicate)
      : ternary_predicate_(std::move(ternary_predicate)) {}

  ~CachedTypeCheck() {
    mutex_lock l(type_to_sequence_map_mu_);
    for (const auto& pair : type_to_sequence_map_) {
      Py_DECREF(pair.first);
    }
  }

  // Caches successful executions of the one-argument (PyObject*) callable
  // "ternary_predicate" based on the type of "o". -1 from the callable
  // indicates an unsuccessful check (not cached), 0 indicates that "o"'s type
  // does not match the predicate, and 1 indicates that it does. Used to avoid
  // calling back into Python for expensive isinstance checks.
  int CachedLookup(PyObject* o) {
    // Try not to return to Python - see if the type has already been seen
    // before.

    auto* type = Py_TYPE(o);

    {
      mutex_lock l(type_to_sequence_map_mu_);
      auto it = type_to_sequence_map_.find(type);
      if (it != type_to_sequence_map_.end()) {
        return it->second;
      }
    }

    int check_result = ternary_predicate_(o);

    if (check_result == -1) {
      return -1;  // Type check error, not cached.
    }

    // NOTE: This is never decref'd as long as the object lives, which is likely
    // forever, but we don't want the type to get deleted as long as it is in
    // the map. This should not be too much of a leak, as there should only be a
    // relatively small number of types in the map, and an even smaller number
    // that are eligible for decref. As a precaution, we limit the size of the
    // map to 1024.
    {
      mutex_lock l(type_to_sequence_map_mu_);
      if (type_to_sequence_map_.size() < kMaxItemsInCache) {
        Py_INCREF(type);
        type_to_sequence_map_.insert({type, check_result});
      }
    }

    return check_result;
  }

 private:
  std::function<int(PyObject*)> ternary_predicate_;
  mutex type_to_sequence_map_mu_;
  std::unordered_map<PyTypeObject*, bool> type_to_sequence_map_
      GUARDED_BY(type_to_sequence_map_mu_);
};

// Returns 1 if `o` is considered a mapping for the purposes of Flatten().
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsMappingHelper(PyObject* o) {
  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return PyObject_IsInstance(to_check, CollectionsMappingType);
  });
  if (PyDict_Check(o)) return true;
  if (TF_PREDICT_FALSE(CollectionsMappingType == nullptr)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        tensorflow::strings::StrCat(
            "collections.Mapping type has not been set. "
            "Please call RegisterMappingClass before using this module")
            .c_str());
    return -1;
  }
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is considered a sequence for the purposes of Flatten().
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsSequenceHelper(PyObject* o) {
  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    int is_instance = PyObject_IsInstance(to_check, CollectionsSequenceType);

    // Don't cache a failed is_instance check.
    if (is_instance == -1) return -1;

    return static_cast<int>(is_instance != 0 && !IsString(to_check));
  });
  // We treat dicts and other mappings as special cases of sequences.
  if (IsMappingHelper(o)) return true;
  if (PySet_Check(o) && !WarnedThatSetIsNotSequence) {
    LOG(WARNING) << "Sets are not currently considered sequences, "
                    "but this may change in the future, "
                    "so consider avoiding using them.";
    WarnedThatSetIsNotSequence = true;
  }
  if (TF_PREDICT_FALSE(CollectionsSequenceType == nullptr)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        tensorflow::strings::StrCat(
            "collections.Sequence type has not been set. "
            "Please call RegisterSequenceClass before using this module")
            .c_str());
    return -1;
  }
  return check_cache->CachedLookup(o);
}

// Implements the same idea as tensorflow.util.nest._yield_value
// During construction we check if the iterable is a dictionary.
// If so, we construct a sequence from its sorted keys that will be used
// for iteration.
// If not, we construct a sequence directly from the iterable.
// At each step, we get the next element from the sequence and use it
// either as a key or return it directly.
//
// 'iterable' must not be modified while ValIterator is used.
class ValIterator {
 public:
  explicit ValIterator(PyObject* iterable)
      : dict_(nullptr),
        mapping_(nullptr),
        last_mapping_element_(nullptr),
        seq_(nullptr),
        index_(0) {
    if (PyDict_Check(iterable)) {
      dict_ = iterable;
      // PyDict_Keys returns a list, which can be used with
      // PySequence_Fast_GET_ITEM.
      seq_ = PyDict_Keys(iterable);
      // Iterate through dictionaries in a deterministic order by sorting the
      // keys. Notice this means that we ignore the original order of
      // `OrderedDict` instances. This is intentional, to avoid potential
      // bugs caused by mixing ordered and plain dicts (e.g., flattening
      // a dict but using a corresponding `OrderedDict` to pack it back).
      PyList_Sort(seq_);
    } else if (IsMappingHelper(iterable)) {
      mapping_ = iterable;
      seq_ = MappingKeys(iterable);
      PyList_Sort(seq_);
    } else {
      seq_ = PySequence_Fast(iterable, "");
    }
    size_ = PySequence_Fast_GET_SIZE(seq_);
  }

  ~ValIterator() { Py_DECREF(seq_); }

  // Return a borrowed reference to the next element from iterable.
  // Return nullptr when iteration is over.
  PyObject* next() {
    if (TF_PREDICT_FALSE(seq_ == nullptr)) {
      return nullptr;
    }
    PyObject* element = nullptr;
    if (index_ < size_) {
      // Both PySequence_Fast_GET_ITEM and PyDict_GetItem return borrowed
      // references. For general mappings, ValIterator keeps a reference to the
      // last retrieved element (and decrefs it before producing the next
      // element) to abstract away the borrowed/new difference.
      element = PySequence_Fast_GET_ITEM(seq_, index_);
      ++index_;
      if (dict_ != nullptr) {
        element = PyDict_GetItem(dict_, element);
        if (element == nullptr) {
          PyErr_SetString(PyExc_RuntimeError,
                          "Dictionary was modified during iteration over it");
          return nullptr;
        }
      } else if (mapping_ != nullptr) {
        element = PyObject_GetItem(mapping_, element);
        if (element == nullptr) {
          PyErr_SetString(PyExc_RuntimeError,
                          "Mapping was modified during iteration over it");
          return nullptr;
        }
        last_mapping_element_.reset(element);
      }
    }
    return element;
  }

 private:
  // Special casing for things that pass PyDict_Check (faster, no Python calls)
  PyObject* dict_;

  // General mappings which have custom Python logic
  PyObject* mapping_;
  Safe_PyObjectPtr last_mapping_element_;

  PyObject* seq_;
  Py_ssize_t size_;
  Py_ssize_t index_;
};

bool IsSparseTensorValueType(PyObject* o) {
  if (TF_PREDICT_FALSE(SparseTensorValueType == nullptr)) {
    return false;
  }

  return PyObject_TypeCheck(o, SparseTensorValueType) == 1;
}

int IsSequenceForDataHelper(PyObject* o) {
  return IsSequenceHelper(o) == 1 && !PyList_Check(o) &&
         !IsSparseTensorValueType(o);
}

bool GetNextValuesForDict(PyObject* nested,
                          std::vector<Safe_PyObjectPtr>* next_values) {
  Safe_PyObjectPtr keys(PyDict_Keys(nested));
  if (PyList_Sort(keys.get()) == -1) return false;
  Py_ssize_t size = PyList_Size(keys.get());
  for (Py_ssize_t i = 0; i < size; ++i) {
    // We know that key and item will not be deleted because nested owns
    // a reference to them and callers of flatten must not modify nested
    // while the method is running.
    PyObject* key = PyList_GET_ITEM(keys.get(), i);
    PyObject* item = PyDict_GetItem(nested, key);
    Py_INCREF(item);
    next_values->emplace_back(item);
  }
  return true;
}

bool GetNextValuesForMapping(PyObject* nested,
                             std::vector<Safe_PyObjectPtr>* next_values) {
  Safe_PyObjectPtr keys(MappingKeys(nested));
  if (keys.get() == nullptr) {
    return false;
  }
  if (PyList_Sort(keys.get()) == -1) return false;
  Py_ssize_t size = PyList_Size(keys.get());
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* key = PyList_GET_ITEM(keys.get(), i);
    // Unlike PyDict_GetItem, PyObject_GetItem returns a new reference.
    PyObject* item = PyObject_GetItem(nested, key);
    next_values->emplace_back(item);
  }
  return true;
}

bool GetNextValuesForIterable(PyObject* nested,
                              std::vector<Safe_PyObjectPtr>* next_values) {
  PyObject* item;
  PyObject* iterator = PyObject_GetIter(nested);
  if (iterator == nullptr || PyErr_Occurred()) {
    return false;
  }
  while ((item = PyIter_Next(iterator)) != nullptr) {
    next_values->emplace_back(item);
  }
  Py_DECREF(iterator);
  return true;
}

// GetNextValues returns the values that the FlattenHelper function will recurse
// over next.
bool GetNextValues(PyObject* nested,
                   std::vector<Safe_PyObjectPtr>* next_values) {
  if (PyDict_Check(nested)) {
    // if nested is dictionary, sort it by key and recurse on each value
    return GetNextValuesForDict(nested, next_values);
  } else if (IsMappingHelper(nested)) {
    // same treatment as dictionaries, but for custom mapping types
    return GetNextValuesForMapping(nested, next_values);
  }
  // iterate and recurse
  return GetNextValuesForIterable(nested, next_values);
}

// Similar to above, just specialized for the functions in the data pacakage.
bool GetNextValuesForData(PyObject* nested,
                          std::vector<Safe_PyObjectPtr>* next_values) {
  if (PyDict_Check(nested)) {
    // if nested is dictionary, sort it by key and recurse on each value
    return GetNextValuesForDict(nested, next_values);
  } else if (IsMappingHelper(nested)) {
    // same treatment as dictionaries, but for custom mapping types
    return GetNextValuesForMapping(nested, next_values);
  } else if (IsSparseTensorValueType(nested)) {
    // if nested is a SparseTensorValue, just return itself as a single item
    Py_INCREF(nested);
    next_values->emplace_back(nested);
    return true;
  }
  // iterate and recurse
  return GetNextValuesForIterable(nested, next_values);
}

bool FlattenHelper(
    PyObject* nested, PyObject* list,
    const std::function<int(PyObject*)>& is_sequence_helper,
    const std::function<bool(PyObject*, std::vector<Safe_PyObjectPtr>*)>&
        next_values_getter) {
  // if nested is not a sequence, append itself and exit
  int is_seq = is_sequence_helper(nested);
  if (is_seq == -1) return false;
  if (!is_seq) {
    return PyList_Append(list, nested) != -1;
  }

  std::vector<Safe_PyObjectPtr> next_values;
  // Get the next values to recurse over.
  if (!next_values_getter(nested, &next_values)) return false;

  for (const auto& item : next_values) {
    if (Py_EnterRecursiveCall(" in flatten")) {
      return false;
    }
    const bool success =
        FlattenHelper(item.get(), list, is_sequence_helper, next_values_getter);
    Py_LeaveRecursiveCall();
    if (!success) {
      return false;
    }
  }
  return true;
}

// Sets error using keys of 'dict1' and 'dict2'.
// 'dict1' and 'dict2' are assumed to be Python dictionaries.
void SetDifferentKeysError(PyObject* dict1, PyObject* dict2, string* error_msg,
                           bool* is_type_error) {
  Safe_PyObjectPtr k1(MappingKeys(dict1));
  if (PyErr_Occurred() || k1.get() == nullptr) {
    *error_msg =
        ("The two dictionaries don't have the same set of keys. Failed to "
         "fetch keys.");
    return;
  }
  Safe_PyObjectPtr k2(MappingKeys(dict2));
  if (PyErr_Occurred() || k2.get() == nullptr) {
    *error_msg =
        ("The two dictionaries don't have the same set of keys. Failed to "
         "fetch keys.");
    return;
  }
  *is_type_error = false;
  *error_msg = tensorflow::strings::StrCat(
      "The two dictionaries don't have the same set of keys. "
      "First structure has keys ",
      PyObjectToString(k1.get()), ", while second structure has keys ",
      PyObjectToString(k2.get()));
}

// Returns true iff there were no "internal" errors. In other words,
// errors that has nothing to do with structure checking.
// If an "internal" error occurred, the appropriate Python error will be
// set and the caller can propage it directly to the user.
//
// Both `error_msg` and `is_type_error` must be non-null. `error_msg` must
// be empty.
// Leaves `error_msg` empty if structures matched. Else, fills `error_msg`
// with appropriate error and sets `is_type_error` to true iff
// the error to be raised should be TypeError.
bool AssertSameStructureHelper(PyObject* o1, PyObject* o2, bool check_types,
                               string* error_msg, bool* is_type_error) {
  DCHECK(error_msg);
  DCHECK(is_type_error);
  const bool is_seq1 = IsSequence(o1);
  const bool is_seq2 = IsSequence(o2);
  if (PyErr_Occurred()) return false;
  if (is_seq1 != is_seq2) {
    string seq_str = is_seq1 ? PyObjectToString(o1) : PyObjectToString(o2);
    string non_seq_str = is_seq1 ? PyObjectToString(o2) : PyObjectToString(o1);
    *is_type_error = false;
    *error_msg = tensorflow::strings::StrCat(
        "Substructure \"", seq_str, "\" is a sequence, while substructure \"",
        non_seq_str, "\" is not");
    return true;
  }

  // Got to scalars, so finished checking. Structures are the same.
  if (!is_seq1) return true;

  if (check_types) {
    const PyTypeObject* type1 = o1->ob_type;
    const PyTypeObject* type2 = o2->ob_type;

    // We treat two different namedtuples with identical name and fields
    // as having the same type.
    const PyObject* o1_tuple = IsNamedtuple(o1, true);
    if (o1_tuple == nullptr) return false;
    const PyObject* o2_tuple = IsNamedtuple(o2, true);
    if (o2_tuple == nullptr) {
      Py_DECREF(o1_tuple);
      return false;
    }
    bool both_tuples = o1_tuple == Py_True && o2_tuple == Py_True;
    Py_DECREF(o1_tuple);
    Py_DECREF(o2_tuple);

    if (both_tuples) {
      const PyObject* same_tuples = SameNamedtuples(o1, o2);
      if (same_tuples == nullptr) return false;
      bool not_same_tuples = same_tuples != Py_True;
      Py_DECREF(same_tuples);
      if (not_same_tuples) {
        *is_type_error = true;
        *error_msg = tensorflow::strings::StrCat(
            "The two namedtuples don't have the same sequence type. "
            "First structure ",
            PyObjectToString(o1), " has type ", type1->tp_name,
            ", while second structure ", PyObjectToString(o2), " has type ",
            type2->tp_name);
        return true;
      }
    } else if (type1 != type2
               /* If both sequences are list types, don't complain. This allows
                  one to be a list subclass (e.g. _ListWrapper used for
                  automatic dependency tracking.) */
               && !(PyList_Check(o1) && PyList_Check(o2))
               /* Two mapping types will also compare equal, making _DictWrapper
                  and dict compare equal. */
               && !(IsMappingHelper(o1) && IsMappingHelper(o2))) {
      *is_type_error = true;
      *error_msg = tensorflow::strings::StrCat(
          "The two namedtuples don't have the same sequence type. "
          "First structure ",
          PyObjectToString(o1), " has type ", type1->tp_name,
          ", while second structure ", PyObjectToString(o2), " has type ",
          type2->tp_name);
      return true;
    }

    if (PyDict_Check(o1) && PyDict_Check(o2)) {
      if (PyDict_Size(o1) != PyDict_Size(o2)) {
        SetDifferentKeysError(o1, o2, error_msg, is_type_error);
        return true;
      }

      PyObject* key;
      Py_ssize_t pos = 0;
      while (PyDict_Next(o1, &pos, &key, nullptr)) {
        if (PyDict_GetItem(o2, key) == nullptr) {
          SetDifferentKeysError(o1, o2, error_msg, is_type_error);
          return true;
        }
      }
    } else if (IsMappingHelper(o1)) {
      // Fallback for custom mapping types. Instead of using PyDict methods
      // which stay in C, we call iter(o1).
      if (PyMapping_Size(o1) != PyMapping_Size(o2)) {
        SetDifferentKeysError(o1, o2, error_msg, is_type_error);
        return true;
      }

      Safe_PyObjectPtr iter(PyObject_GetIter(o1));
      PyObject* key;
      while ((key = PyIter_Next(iter.get())) != nullptr) {
        if (!PyMapping_HasKey(o2, key)) {
          SetDifferentKeysError(o1, o2, error_msg, is_type_error);
          Py_DECREF(key);
          return true;
        }
        Py_DECREF(key);
      }
    }
  }

  ValIterator iter1(o1);
  ValIterator iter2(o2);

  while (true) {
    PyObject* v1 = iter1.next();
    PyObject* v2 = iter2.next();
    if (v1 != nullptr && v2 != nullptr) {
      if (Py_EnterRecursiveCall(" in assert_same_structure")) {
        return false;
      }
      bool no_internal_errors = AssertSameStructureHelper(
          v1, v2, check_types, error_msg, is_type_error);
      Py_LeaveRecursiveCall();
      if (!no_internal_errors) return false;
      if (!error_msg->empty()) return true;
    } else if (v1 == nullptr && v2 == nullptr) {
      // Done with all recursive calls. Structure matched.
      return true;
    } else {
      *is_type_error = false;
      *error_msg = tensorflow::strings::StrCat(
          "The two structures don't have the same number of elements. ",
          "First structure: ", PyObjectToString(o1),
          ". Second structure: ", PyObjectToString(o2));
      return true;
    }
  }
}

}  // namespace

void RegisterSequenceClass(PyObject* sequence_class) {
  if (!PyType_Check(sequence_class)) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Expecting a class definition for `collections.Sequence`. Got ",
            Py_TYPE(sequence_class)->tp_name)
            .c_str());
    return;
  }
  CollectionsSequenceType = sequence_class;
}

void RegisterMappingClass(PyObject* mapping_class) {
  if (!PyType_Check(mapping_class)) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Expecting a class definition for `collections.Mapping`. Got ",
            Py_TYPE(mapping_class)->tp_name)
            .c_str());
    return;
  }
  CollectionsMappingType = mapping_class;
}

void RegisterSparseTensorValueClass(PyObject* sparse_tensor_value_class) {
  if (!PyType_Check(sparse_tensor_value_class)) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Expecting a class definition for `SparseTensorValue`. Got ",
            Py_TYPE(sparse_tensor_value_class)->tp_name)
            .c_str());
    return;
  }
  SparseTensorValueType =
      reinterpret_cast<PyTypeObject*>(sparse_tensor_value_class);
}

bool IsSequence(PyObject* o) { return IsSequenceHelper(o) == 1; }
bool IsMapping(PyObject* o) { return IsMappingHelper(o) == 1; }

PyObject* Flatten(PyObject* nested) {
  PyObject* list = PyList_New(0);
  if (FlattenHelper(nested, list, IsSequenceHelper, GetNextValues)) {
    return list;
  } else {
    Py_DECREF(list);
    return nullptr;
  }
}

bool IsSequenceForData(PyObject* o) { return IsSequenceForDataHelper(o) == 1; }

PyObject* FlattenForData(PyObject* nested) {
  PyObject* list = PyList_New(0);
  if (FlattenHelper(nested, list, IsSequenceForDataHelper,
                    GetNextValuesForData)) {
    return list;
  } else {
    Py_DECREF(list);
    return nullptr;
  }
}

PyObject* IsNamedtuple(PyObject* o, bool strict) {
  // Must be subclass of tuple
  if (!PyTuple_Check(o)) {
    Py_RETURN_FALSE;
  }

  // If strict, o.__class__.__base__ must be tuple
  if (strict) {
    PyObject* klass = PyObject_GetAttrString(o, "__class__");
    if (klass == nullptr) return nullptr;
    PyObject* base = PyObject_GetAttrString(klass, "__base__");
    Py_DECREF(klass);
    if (base == nullptr) return nullptr;

    const PyTypeObject* base_type = reinterpret_cast<PyTypeObject*>(base);
    // built-in object types are singletons
    bool tuple_base = base_type == &PyTuple_Type;
    Py_DECREF(base);
    if (!tuple_base) {
      Py_RETURN_FALSE;
    }
  }

  if (TF_PREDICT_FALSE(CollectionsSequenceType == nullptr)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        tensorflow::strings::StrCat(
            "collections.Sequence type has not been set. "
            "Please call RegisterSequenceClass before using this module")
            .c_str());
    return nullptr;
  }

  // o must have attribute '_fields' and every element in
  // '_fields' must be a string.
  int has_fields = PyObject_HasAttrString(o, "_fields");
  if (!has_fields) {
    Py_RETURN_FALSE;
  }

  Safe_PyObjectPtr fields = make_safe(PyObject_GetAttrString(o, "_fields"));
  int is_instance = PyObject_IsInstance(fields.get(), CollectionsSequenceType);
  if (is_instance == 0) {
    Py_RETURN_FALSE;
  } else if (is_instance == -1) {
    return nullptr;
  }

  Safe_PyObjectPtr seq = make_safe(PySequence_Fast(fields.get(), ""));
  const Py_ssize_t s = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < s; ++i) {
    // PySequence_Fast_GET_ITEM returns borrowed ref
    PyObject* elem = PySequence_Fast_GET_ITEM(seq.get(), i);
    if (!IsString(elem)) {
      Py_RETURN_FALSE;
    }
  }

  Py_RETURN_TRUE;
}

PyObject* SameNamedtuples(PyObject* o1, PyObject* o2) {
  PyObject* f1 = PyObject_GetAttrString(o1, "_fields");
  PyObject* f2 = PyObject_GetAttrString(o2, "_fields");
  if (f1 == nullptr || f2 == nullptr) {
    Py_XDECREF(f1);
    Py_XDECREF(f2);
    PyErr_SetString(
        PyExc_RuntimeError,
        "Expected namedtuple-like objects (that have _fields attr)");
    return nullptr;
  }

  if (PyObject_RichCompareBool(f1, f2, Py_NE)) {
    Py_RETURN_FALSE;
  }

  if (GetClassName(o1).compare(GetClassName(o2)) == 0) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject* AssertSameStructure(PyObject* o1, PyObject* o2, bool check_types) {
  string error_msg;
  bool is_type_error = false;
  AssertSameStructureHelper(o1, o2, check_types, &error_msg, &is_type_error);
  if (PyErr_Occurred()) {
    // Don't hide Python exceptions while checking (e.g. errors fetching keys
    // from custom mappings).
    return nullptr;
  }
  if (!error_msg.empty()) {
    PyErr_SetString(
        is_type_error ? PyExc_TypeError : PyExc_ValueError,
        tensorflow::strings::StrCat(
            "The two structures don't have the same nested structure.\n\n",
            "First structure: ", PyObjectToString(o1), "\n\nSecond structure: ",
            PyObjectToString(o2), "\n\nMore specifically: ", error_msg)
            .c_str());
    return nullptr;
  }
  Py_RETURN_NONE;
}

}  // namespace swig
}  // namespace tensorflow
