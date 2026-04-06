class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) model.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int = 3,
        hidden: tuple[int, ...] = HIDDEN_LAYERS,
        pdrop: float = 0.1,
    ):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(pdrop)]
            last = h
        layers += [nn.Linear(last, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPWithSymbol(nn.Module):
    """
    Multi-layer perceptron (MLP) model with symbol embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        n_symbols: int,
        emb_dim: int = 16,
        n_classes: int = 3,
        hidden: tuple[int, ...] = (256, 128, 64),
        pdrop: float = 0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=n_symbols, embedding_dim=emb_dim)
        self.mlp = MLP(in_dim + emb_dim, n_classes=n_classes, hidden=hidden, pdrop=pdrop)

    def forward(self, x_num, sym_id):
        e = self.emb(sym_id)
        z = torch.cat([x_num, e], dim=1)
        return self.mlp(z)


def main(args):
    # Fit normalization from TRAIN only
    x_tr = train_df[numeric_cols].astype(np.float32).values
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
    std[std == 0] = 1.0
    for ds in (tr_ds, va_ds, te_ds):
        ds.set_norm(mean, std)

    # Loaders
    kwargs = dict(batch_size=args.batch_size, num_workers=2, pin_memory=(device.type == "cuda"))
    if args.use_emb:
        pass  # default works since each __getitem__ returns (x_num, sym_id, y)
    else:
        pass
    tr_ld = DataLoader(tr_ds, shuffle=True, **kwargs)
    va_ld = DataLoader(va_ds, shuffle=False, **kwargs)
    te_ld = DataLoader(te_ds, shuffle=False, **kwargs)

    # Class weights from TRAIN only
    class_weights = compute_class_weights(train_df[target_col].values, n_classes=3)

    # Model
    if args.use_emb:
        model = MLPWithSymbol(
            in_dim=len(numeric_cols),
            n_symbols=len(sym2id),
            emb_dim=args.emb_dim,
            n_classes=3,
            hidden=(args.h1, args.h2, args.h3),
            pdrop=args.dropout,
        )
        use_emb = True
    else:
        model = MLP(
            in_dim=len(numeric_cols),
            n_classes=3,
            hidden=(args.h1, args.h2, args.h3),
            pdrop=args.dropout,
        )
        use_emb = False

    # Train
    model = train_loop(
        model, tr_ld, va_ld, device, class_weights, epochs=args.epochs, lr=args.lr, use_emb=use_emb
    )

    # Evaluate on TEST
    macro_f1, y_true, y_pred = evaluate(model, te_ld, device, use_emb)
    print("\nTEST macro-F1:", round(macro_f1, 4))
    print("\nClassification report (TEST):\n")
    print(classification_report(y_true, y_pred, digits=4))

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "use_emb": use_emb,
            "sym2id": sym2id,
            "numeric_cols": numeric_cols,
            "mean": mean,
            "std": std,
            "args": vars(args),
        },
        out_dir / "model.pt",
    )
    print(f"Saved model to {out_dir / 'model.pt'}")
