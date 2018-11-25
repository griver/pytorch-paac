import json
from collections import namedtuple


class RoomData(object):
    __slots__ = ['id', 'name', 'entry_dist', 'spawn_spots', 'doors', 'texture']

    def __init__(self, id, name, entry_dist,
                 spawn_spots=tuple(), doors=tuple(), texture=0):
        self.id = id
        self.name = name
        self.entry_dist = entry_dist # Right now entry_dist just equals the number of doors between the entry room and this room
        self.spawn_spots = spawn_spots #A list of spawn spots on a way from the entry room to this room
        self.doors = doors #A list of doors on a way from the entry room to this room
        self.texture = texture

    def __repr__(self):
        return "RoomData({r.id},{r.name},dst={r.entry_dist}," \
               "spots={r.spawn_spots},doors={r.doors},texture={r.texture})".format(r=self)

    @classmethod
    def serialize_to_json(cls, obj):
        d = { '__classname__': cls.__name__}
        d.update({f:getattr(obj,f) for f in obj.__slots__})
        return d

    @classmethod
    def unserialize_from_json(cls, d):
        clsname = d.pop('__classname__', None)
        if clsname == cls.__name__:
            return cls(**d)
        else:
            return d


def load_map_info(file_path, ext='.json'):
    def unserialize_map_info(d):
        if isinstance(d, dict):
            if '__classname__' in d:
                return RoomData.unserialize_from_json(d)
            elif all(k.isdecimal() for k in d.keys()):
                return {int(k):v for k,v in d.items()}
        return d

    with open(file_path+ext, 'r') as f:
        data = json.load(f, object_hook=unserialize_map_info)

    return data


def create_json_config():
    import json

    def save_json(file_path, data):
        with open(file_path, 'w') as f:
            return json.dump(data, f, default=RoomData.serialize_to_json, sort_keys=True)

    data = {}
    data['items'] = {
        0:"BlueSkullItem", 1:"YellowSkullItem",
        2:"GreenColumnItem", 3:"BerserkItem"
    }
    #{"STONE2", "REDC1", "GREC1", "BLUC1", "YELC1", "BLAC1", "WHIC1", "ZZWOLF11", "ZZWOLF13"}
    data['textures'] = ((0, 'stone wall'), (1,'red mark'), (2, 'green mark'),
                        (3, 'blue mark'), (4, 'yellow mark'), (5, 'black mark'),
                        (6, 'white mark'),(7, "red brick wall"), (8, 'red wall with gray shield'))
    data['default_texture_id'] = 0
    data['entry_room'] = RoomData(9, 'entry_room', 0, spawn_spots=(90,91,92), doors=())
    data["rooms"] = {
        9:RoomData(9, 'entry_room', 0, spawn_spots=(90,91,92), doors=()),
        11:RoomData(11, 'room11', 1, spawn_spots=(110,), doors=(110,)),
        12:RoomData(12, 'room12', 1, spawn_spots=(120,), doors=(120,)),
        13:RoomData(13, 'room13', 1, spawn_spots=(130,), doors=(130,)),
        14:RoomData(14, 'room14', 1, spawn_spots=(140,), doors=(140,)),
        21:RoomData(21, 'room21', 2, spawn_spots=(210,110), doors=(210,110)),
        22:RoomData(22, 'room22', 2, spawn_spots=(220, 120), doors=(220, 120)),
        23:RoomData(23, 'room23', 2, spawn_spots=(230, 130), doors=(230, 130)),
        24:RoomData(24, 'room24', 2, spawn_spots=(240, 140), doors=(240, 140)),
    }
    data["spawn_spots"] = [90,91,92,110,120,130,140,210,220,230,240]

    save_json('resources/vizdoom_scenarios/warehouse.json',data)
