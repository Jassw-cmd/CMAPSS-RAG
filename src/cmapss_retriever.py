from load_data import load_cmapss

def get_engine_data(data, engine_id):
    engine = data[data["engine_id"] == engine_id].copy()
    if engine.empty:
        raise ValueError(f"engine {engine_id} was not found.")
    return engine.sort_values("cycle").reset_index(drop = True)

def get_sensor_value(data, engine_id, cycle, sensor_id):
    if not 1 <= sensor_id <= 21:
        raise ValueError("sensor_id must be between  1 and 21")
    engine = get_engine_data(data, engine_id)
    row = engine[engine["cycle"] == cycle]
    if row.empty:
        raise ValueError(f"Cycle {cycle} was not found for engine {engine_id}")

    return float(row.iloc[0][f"sensor_{sensor_id}"])

def get_rul(data, engine_id, cycle=None):
    """Calculate training-set RUL for a specific cycle."""
    engine = get_engine_data(data, engine_id)

    if cycle is None:
        return {
            "status": "missing_information",
            "message": (
                f"At which cycle do you want to calculate "
                f"the RUL of Engine {engine_id}?"
            ),
        }

    if cycle not in engine["cycle"].values:
        raise ValueError(
            f"Cycle {cycle} was not found for engine {engine_id}."
        )

    final_cycle = int(engine["cycle"].max())

    return {
        "status": "success",
        "engine_id": engine_id,
        "cycle": cycle,
        "rul": final_cycle - cycle,
    }


def main():
    data = load_cmapss()

    print("Dataset shape:", data.shape)

    # Test 1: Query RUL without specifying a cycle
    print("\n--- Test 1: RUL without cycle ---")
    result = get_rul(data, engine_id=24)
    print(result["message"])

    # Test 2: Query RUL with a specific cycle
    print("\n--- Test 2: RUL at cycle 100 ---")
    result = get_rul(data, engine_id=24, cycle=100)
    print(result)

    # Test 3: Query a sensor value
    print("\n--- Test 3: Sensor value ---")
    try:
        sensor_id = 7
        value = get_sensor_value(
            data,
            engine_id=24,
            cycle=100,
            sensor_id=sensor_id,
        )
        print(f"Engine 24, cycle 100, sensor {sensor_id}: {value}")
    except ValueError as error:
        print("Query error:", error)

    # Test 4: Find the final observed cycle
    print("\n--- Test 4: Final observed cycle ---")
    try:
        engine = get_engine_data(data, engine_id=24)
        final_cycle = int(engine["cycle"].max())
        print(f"Engine 24 final observed cycle: {final_cycle}")
    except ValueError as error:
        print("Query error:", error)

    # Test 5: Query a nonexistent engine
    print("\n--- Test 5: Invalid engine ---")
    try:
        result = get_rul(data, engine_id=999, cycle=100)
        print(result)
    except ValueError as error:
        print("Query error:", error)


if __name__ == "__main__":
    main()