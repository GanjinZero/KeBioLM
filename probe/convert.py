def convert(relation):
    relation = relation.replace("_", " ").replace("isa", "is a")
    if len(relation) > 2:
        if relation.split(" ")[-1] in ["of", "as", "by"]:
            relation = "is " + relation
    return relation