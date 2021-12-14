import toml

if __name__ == '__main__':
    rext = """
    [feature]
    [[feature.towers]]
    name="user"
    [[feature.towers.fields]]
    name="age"
    pad=0
    [[feature.towers.fields]]
    name="height"
    pad=0
    
    [[feature.towers]]
    name="sku"
    [[feature.towers.fields]]
    name="sku_id"
    pad=0
    """

    res = toml.loads(rext)
    print(res)