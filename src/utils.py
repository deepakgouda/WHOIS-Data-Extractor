from typing import List, Dict, Tuple, Union, Optional

def entity_to_bio_labels(entities: List[str]):
    bio_labels = ["O"] + ["%s-%s" % (bi, label) for label in entities for bi in "BI"]
    return bio_labels

def span_list_to_dict(span_list: List[list]) -> Dict[Tuple[int, int], Union[str, tuple]]:
    """
    convert entity label span list to span dictionaries

    Parameters
    ----------
    span_list

    Returns
    -------
    span_dict
    """
    span_dict = dict()
    for span in span_list:
        span_dict[(span[0], span[1])] = span[2]
    return span_dict

def span_to_label(labeled_spans: Dict[Tuple[int, int], str], tokens: List[str]) -> List[str]:
    """
    Convert entity spans to labels

    Parameters
    ----------
    labeled_spans: labeled span dictionary: {(start, end): label}
    tokens: a list of tokens, used to check if the spans are valid.

    Returns
    -------
    a list of string labels
    """
    if labeled_spans:
        assert list(labeled_spans.keys())[-1][1] <= len(tokens), ValueError("label spans out of scope!")

    labels = ["O"] * len(tokens)
    for (start, end), label in labeled_spans.items():
        if type(label) == list or type(label) == tuple:
            lb = label[0][0]
        else:
            lb = label
        labels[start] = "B-" + lb
        if end - start > 1:
            labels[start + 1 : end] = ["I-" + lb] * (end - start - 1)

    return labels


def pack_instances(**kwargs) -> list[dict]:
    """
    Convert attribute lists to a list of data instances, each is a dict with attribute names as keys
    and one datapoint attribute values as values
    """

    instance_list = list()
    keys = tuple(kwargs.keys())

    for inst_attrs in zip(*tuple(kwargs.values())):
        inst = dict(zip(keys, inst_attrs))
        instance_list.append(inst)

    return instance_list

def unpack_instances(instance_list: list[dict], attr_names: Optional[list[str]] = None):
    """
    Convert a list of dict-type instances to a list of value lists,
    each contains all values within a batch of each attribute

    Parameters
    ----------
    instance_list: list[dict]
        a list of attributes
    attr_names: list[str], optional
        the name of the needed attributes. Notice that this variable should be specified
        for Python versions that does not natively support ordered dict
    """
    if not attr_names:
        attr_names = list(instance_list[0].keys())
    attribute_tuple = [[inst[name] for inst in instance_list] for name in attr_names]

    return attribute_tuple