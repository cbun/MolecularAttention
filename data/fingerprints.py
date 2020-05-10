def load_fingerprint_data(file_path, header="ecfp4_512", nrows=None):

    try:
        df = pd.read_csv(
            file_path,
            sep="\t",
            nrows=nrows,
            dtype={
                "canonical_smile": str,
                "smile": str,
                "dock_score": float,
                "maccs_key": str,
                "ecfp2_512": str,
                "ecfp4_512": str,
                "ecfp6_512": str,
                "ecfp2_2048": str,
                "ecfp4_2048": str,
                "ecfp6_2048": str,
            },
        )

        print("Fingerprints info:", df.info())
    except Exception as e:
        print("loading data failure", e)
        exit()

    df = df.fillna(0)
    smiles = df["canonical_smile"]
    docking_score = abs(np.clip(df["dock_score"], a_min=None, a_max=0))

    fingerprint = np.array(df[header])
    docking_score = np.asarray(docking_score).astype(np.float32)

    fingerprint_raw_temp = np.asarray(fingerprint)
    fingerprint_raw = []

    for item in fingerprint_raw_temp:
        item_ = (
            item.replace("[", ",").replace("]", ",").replace(" ", "").split(",")[1:-1]
        )
        fingerprint_raw.append(np.asarray(item_).astype(np.float32))

    fingerprint_raw = np.asarray(fingerprint_raw)
    return smiles, fingerprint_raw, docking_score
