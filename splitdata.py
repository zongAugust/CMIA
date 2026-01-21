def process_file(input_file):
    # read
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_users = len(lines)
    half = total_users // 2
    shadow_users = lines[:half]
    target_users = lines[half:]

    # count
    all_items = set()
    total_interactions = 0

    shadow_items = set()
    shadow_interactions = 0

    target_items = set()
    target_interactions = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        items = parts[1:]
        total_interactions += len(items)
        all_items.update(items)

    # shadow data
    with open("../gowalla/attackdata/shadowtest.txt", "w") as shadow_f:
        for line in shadow_users:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            items = parts[1:]
            shadow_interactions += len(items)
            shadow_items.update(items)
            shadow_f.write(line + '\n')

    # target data
    with open("../gowalla/attackdata/targettest.txt", "w") as target_f:
        for new_uid, line in enumerate(target_users):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            items = parts[1:]
            target_interactions += len(items)
            target_items.update(items)
            new_line = str(new_uid) + ' ' + ' '.join(items)
            target_f.write(new_line + '\n')

    print("==========")
    print(f"ori data:")
    print(f"user nums:{total_users}")
    print(f"item nums:{len(all_items)}")
    print(f"interactions:{total_interactions}")

    print(f"\nShadow")
    print(f"user nums:{len(shadow_users)}")
    print(f"item nums:{len(shadow_items)}")
    print(f"interactions:{shadow_interactions}")

    print(f"\nTarget")
    print(f"user nums:{len(target_users)}")
    print(f"item nums:{len(target_items)}")
    print(f"interactions:{target_interactions}")
    print("===========================")

if __name__ == "__main__":
    process_file("../gowalla/test.txt")