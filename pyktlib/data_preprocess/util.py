import pandas as pd


def map_user_info(df, field):
    # 将用户的指定信息进行重映射，并按照用户数量排序（如重映射学校id，那么学生数量大的学校id被映射为1，其它学校依次映射为2,3···）
    num_user_in_field = df[df[field] != -1].groupby(field).agg(user_count=("user_id", lambda x: x.nunique())).to_dict()
    num_user_in_field = list(num_user_in_field["user_count"].items())
    num_user_in_field = sorted(num_user_in_field, key=lambda item: item[1], reverse=True)
    field_id_map = {item[0]: i+1 for i, item in enumerate(num_user_in_field)}
    field_id_map[-1] = 0
    df[field] = df[field].map(field_id_map)

    num_user_in_field = list(map(lambda item: (field_id_map[item[0]], item[1]), num_user_in_field))
    field_id_map = pd.DataFrame({
        field: list(field_id_map.keys()),
        f"{field}_map": list(field_id_map.values())
    })
    field_info = {field_id: {
        "num_user": num_user,
        "num_interaction": len(df[df[field] == field_id])
    } for field_id, num_user in num_user_in_field}
    field_info[-1] = {
        "num_user": len(pd.unique(df[df[field] == -1]["user_id"])),
        "num_interaction": len(df[df[field] == -1])
    }

    return field_id_map, field_info
