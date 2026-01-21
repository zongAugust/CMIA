import random

def generate_nonmember_and_attack_files(shadow_file, nonmember_file, attack_file):
    random.seed(42)  

    user2items = {}
    all_items = set()

    # Step 1
    with open(shadow_file, 'r') as fin:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2:
                continue 
            user = parts[0]
            items = list(map(int, parts[1:]))
            user2items[user] = set(items)
            all_items.update(items)

    all_items = list(all_items) 

    # Step 2: shadownonmember.txt and attacktrain.txt
    with open(nonmember_file, 'w') as fnon, open(attack_file, 'w') as fattack:
        for user, pos_items in user2items.items():
            pos_items = list(pos_items)

            # （label=1）
            for item in pos_items:
                fattack.write(f"{user} {item} 1\n")

            neg_candidates = list(set(all_items) - set(pos_items))

            if len(neg_candidates) < len(pos_items):
                sampled_neg_items = random.choices(neg_candidates, k=len(pos_items))
            else:
                sampled_neg_items = random.sample(neg_candidates, len(pos_items))

            # label=0
            fnon.write(f"{user} {' '.join(map(str, sampled_neg_items))}\n")

            for item in sampled_neg_items:
                fattack.write(f"{user} {item} 0\n")

    print(f"Done")

if __name__ == "__main__":
    shadow_file = "path"
    nonmember_file = "path"
    attack_file = "path"

    generate_nonmember_and_attack_files(shadow_file, nonmember_file, attack_file)