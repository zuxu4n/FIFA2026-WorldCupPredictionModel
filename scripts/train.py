"""Train the XGBoost Poisson expected-goals model.

    python scripts/train.py
"""
from wcpred.model import train, feature_importance


def main():
    booster, meta = train(save=True, verbose=True)
    print("\nTop 15 features by gain:")
    for k, v in feature_importance(booster, "gain")[:15]:
        print(f"  {k:22s} {v:10.1f}")


if __name__ == "__main__":
    main()
