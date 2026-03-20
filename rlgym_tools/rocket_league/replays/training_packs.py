import json
import math
import re
import struct
from datetime import datetime

AES_KEY = bytes([
    0xD7, 0x8C, 0x32, 0x4A, 0x94, 0x42, 0x94, 0x3C,
    0x6D, 0x65, 0xCE, 0x98, 0x81, 0x85, 0x4C, 0x41,
    0x68, 0x99, 0x22, 0x0C, 0xC7, 0xA1, 0x46, 0x40,
    0x93, 0x9B, 0x96, 0x3C, 0x93, 0x2A, 0x6F, 0xAF
])

TAGS = {
    0: "Aerials",
    1: "Bounces",
    2: "Shots",
    3: "Saves",
    4: "Clears",
    5: "Rebounds",
    6: "Redirects",
    7: "Freestyles",
    8: "Dribbles",
    9: "Air Dribbles",
    10: "Kickoffs",
    11: "Wall Shots",
    12: "Long Shots",
    13: "Close Shots",
    14: "Angle Shots",
    15: "Backwards Shots",
    16: "Offense",
    17: "Defense",
    18: "Pinch Shots",
}

DIFFICULTIES = {
    "D_Easy": "Rookie",
    "D_Medium": "Pro",
    "D_Hard": "All-Star"
}


def ue_rot_to_rad(rot_int):
    return (rot_int / 65536.0) * (2 * math.pi)


def calculate_lin_vel(speed, pitch_rot, yaw_rot):
    p, y = ue_rot_to_rad(pitch_rot), ue_rot_to_rad(yaw_rot)
    return [
        float(math.cos(p) * math.cos(y) * speed),
        float(math.cos(p) * math.sin(y) * speed),
        float(math.sin(p) * speed)
    ]


def extract_meta_string(data, prop):
    pattern = prop + b'\x00.+?(?:StrProperty|NameProperty|ByteProperty)\x00.{8,24}?([A-Za-z0-9_ -]{3,40})\x00'
    match = re.search(pattern, data, re.DOTALL)
    return match.group(1).decode('ascii', errors='ignore') if match else "Unknown"


def training_pack_to_json(input_path):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    with open(input_path, 'rb') as f:
        length = struct.unpack('<I', f.read(4))[0]
        f.read(4)
        encrypted_data = f.read(length)

    decryptor = Cipher(algorithms.AES(AES_KEY), modes.ECB()).decryptor()
    decrypted = decryptor.update(encrypted_data) + decryptor.finalize()

    # Metadata extraction
    metadata = {
        'title': extract_meta_string(decrypted, b'TM_Name'),
        'code': extract_meta_string(decrypted, b'Code'),
        'author': extract_meta_string(decrypted, b'CreatorName'),
        'map': extract_meta_string(decrypted, b'MapName'),
        'tags': []
    }

    type_match = re.search(b'Type\x00.{4}ByteProperty\x00.{8}.{4}ETrainingType\x00.{4}(.*?)\x00', decrypted, re.DOTALL)
    training_type = type_match.group(1).decode('ascii') if type_match else "Unknown"
    metadata['type'] = training_type.replace("Training_", "")

    diff_match = re.search(b'Difficulty\x00.{4}ByteProperty\x00.{8}.{4}EDifficulty\x00.{4}(.*?)\x00', decrypted,
                           re.DOTALL)
    raw_diff = diff_match.group(1).decode('ascii') if diff_match else "Unknown"
    metadata['difficulty'] = DIFFICULTIES.get(raw_diff, raw_diff)

    time_match = re.search(b'UpdatedAt\x00.+?QWordProperty\x00\x08\x00\x00\x00\x00\x00\x00\x00(.{8})', decrypted,
                           re.DOTALL)
    if time_match:
        metadata['last_updated'] = datetime.fromtimestamp(struct.unpack('<Q', time_match.group(1))[0]).strftime(
            '%Y-%m-%d %H:%M:%S')

    tags_match = re.search(b'Tags\x00.{4}ArrayProperty\x00.{4}\x00\x00\x00\x00(.{4})', decrypted, re.DOTALL)
    if tags_match:
        count = struct.unpack('<I', tags_match.group(1))[0]
        raw_tags = struct.unpack(f'<{count}I', decrypted[tags_match.end(): tags_match.end() + (count * 4)])
        metadata['tags'] = [TAGS.get(t, t) for t in raw_tags]

    # Shot extraction
    shot_blocks = decrypted.split(b'TimeLimit\x00')[1:]
    shots = []

    for block in shot_blocks:
        tl_match = re.search(b'FloatProperty\x00\x04\x00\x00\x00\x00\x00\x00\x00(.{4})', block, re.DOTALL)
        time_limit = struct.unpack('<f', tl_match.group(1))[0] if tl_match else 0.0

        entities = {}
        for blob in re.findall(b'{[^{}]+}', block):
            try:
                obj = json.loads(blob.decode('latin-1', errors='ignore'))
                if 'Ball' in obj.get('ObjectArchetype', ''):
                    entities['ball'] = obj
                elif 'DynamicSpawnPointMesh' in obj.get('ObjectArchetype', ''):
                    entities['car'] = obj
            except:
                continue

        if 'ball' in entities and 'car' in entities:
            b, c = entities['ball'], entities['car']
            shots.append({
                'time_limit': round(time_limit, 2),
                'ball': {
                    'position': [b.get('StartLocationX', 0),
                                 b.get('StartLocationY', 0),
                                 b.get('StartLocationZ', 0)],
                    'linear_velocity': calculate_lin_vel(b.get('VelocityStartSpeed', 0),
                                                         b.get('VelocityStartRotationP', 0),
                                                         b.get('VelocityStartRotationY', 0)),
                    'euler_angles': [0.0, 0.0, 0.0]  # Ball spawns with 0 rotation/angular velocity
                },
                'car': {
                    'position': [c.get('LocationX', 0), c.get('LocationY', 0), c.get('LocationZ', 0)],
                    'linear_velocity': [0.0, 0.0, 0.0],
                    'euler_angles': [ue_rot_to_rad(c.get('RotationP', 0)),
                                     ue_rot_to_rad(c.get('RotationY', 0)),
                                     ue_rot_to_rad(c.get('RotationR', 0))]
                }
            })
        else:
            raise ValueError("Missing ball or car data in shot block")

    return {'metadata': metadata, 'shots': shots}


if __name__ == '__main__':
    import glob
    from collections import Counter

    folder = r"C:\Users\*\Documents\My Games\Rocket League\TAGame\Training\*\Favorities"
    seen_codes = set()
    total_shots = 0
    tags_included = Counter()
    difficulties_included = Counter()
    types_included = Counter()
    for fpath in glob.glob(folder + r"\**\*.Tem", recursive=True):
        data = training_pack_to_json(fpath)
        pack_meta = data['metadata']
        pack_shots = data['shots']

        code = pack_meta['code']
        if code in seen_codes:
            print(f"Duplicate code found, skipping: {code} ({fpath})")
            continue
        seen_codes.add(code)

        print(
            f"Pack: {pack_meta['title']} by {pack_meta['author']} (Type: {pack_meta['type']}, Map: {pack_meta['map']}, Difficulty: {pack_meta['difficulty']})")
        print(f"  Code: {pack_meta['code']}")
        print(f"  Last Updated: {pack_meta.get('last_updated', 'Unknown')}")
        print(f"  Tags: {', '.join(pack_meta['tags'])}")
        print(f"  Shots: {len(pack_shots)}")
        for idx, shot in enumerate(pack_shots, 1):
            print(f"    Shot {idx}: Time Limit: {shot['time_limit']}s, Ball Pos: {shot['ball']['position']}, "
                  f"Car Pos: {shot['car']['position']}, Car Rot: {shot['car']['euler_angles']}")
        print()

        total_shots += len(pack_shots)
        for tag in pack_meta['tags']:
            tags_included[tag] += 1
        difficulties_included[pack_meta['difficulty']] += 1
        types_included[pack_meta['type']] += 1

    tags_not_included = set(TAGS.values()) - set(tags_included.keys())
    print(f"Total Packs: {len(seen_codes)}")
    print(f"Total Shots: {total_shots}")
    print("Included Tags: " + ", ".join(f"{tag} ({count})" for tag, count in tags_included.most_common()))
    if tags_not_included:
        print(f"Tags Not Included: {', '.join(tags_not_included)}")
    print("Included Difficulties: " + ", ".join(
        f"{diff} ({count})" for diff, count in difficulties_included.most_common()))
    print("Included Types: " + ", ".join(f"{t} ({count})" for t, count in types_included.most_common()))
