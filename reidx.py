input_path = "./yelp2018/setting1/targettrain1.txt"            
output_path = "./yelp2018/setting1/re_targettrain1.txt"     

user_old2new = {}
next_id = 0

with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for line in fin:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        user = int(parts[0])
        items = parts[1:]

        # 映射 user id
        if user not in user_old2new:
            user_old2new[user] = next_id
            next_id += 1
        new_user = user_old2new[user]

        # 写入新行
        fout.write(f"{new_user} {' '.join(items)}\n")

print(f"Re-indexed {len(user_old2new)} users. Output written to {output_path}")