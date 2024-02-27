from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter
from torch.nn.init import xavier_uniform_
from tqdm import tqdm

from src.model.SEEK import SEEK
from src.model.common import evaluate, count_parameters, make_infinite
from src.utils import config
from src.utils.common import set_seed
from src.utils.data.loader import prepare_data_seq


def make_model(vocab, dec_num):
    is_eval = config.test
    model = SEEK(
        vocab,
        decoder_number=dec_num,
        is_eval=is_eval,
        model_file_path=config.model_path if is_eval else None,
    )

    model.to(config.device)

    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    print("# PARAMETERS", count_parameters(model))

    return model


def train(model, train_set, dev_set):
    check_iter = 1000
    max_tra_iter = 24000
    try:
        model.train()
        best_ppl = 1000
        best_loss = 1000
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(train_set)
        for n_iter in tqdm(range(1000000)):
            loss, ppl, bce, accs, _, _ = model.train_one_batch(next(data_iter), n_iter)
            (dia_acc, utt_acc, trg_acc) = accs
            acc = utt_acc

            writer.add_scalars("loss", {"loss_train": loss}, n_iter)
            writer.add_scalars("ppl", {"ppl_train": ppl}, n_iter)
            writer.add_scalars("cls_loss", {"cls_loss_train": bce}, n_iter)
            writer.add_scalars("accuracy", {"acc_train": acc}, n_iter)
            if config.noam:
                writer.add_scalars(
                    "lr", {"learning_rata": model.optimizer._rate}, n_iter
                )

            if (n_iter + 1) % check_iter == 0:
                model.eval()
                model.epoch = n_iter
                loss_val, ppl, dia_acc_val, ctx_acc_val, trg_acc_val, results = evaluate(
                    model, dev_set, ty="valid", max_dec_step=50
                )

                model.train()

                if loss_val <= best_loss:
                    print("Loss: {:.4f} dia_acc: {:.4f} ctx_acc: {:.4f} trg_acc: {:.4f}  *"
                          .format(loss_val, dia_acc_val, ctx_acc_val, trg_acc_val))

                    best_loss = loss_val

                    patient = 0
                    model.save_model(best_ppl, n_iter)
                    weights_best = deepcopy(model.state_dict())
                else:
                    print("Loss: {:.4f} dia_acc: {:.4f} ctx_acc: {:.4f} trg_acc: {:.4f}"
                          .format(loss_val, dia_acc_val, ctx_acc_val, trg_acc_val))
                    patient += 1
                if n_iter < max_tra_iter:
                    continue
                if patient > 6:
                    break

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
        model.save_model(best_loss, n_iter)
        weights_best = deepcopy(model.state_dict())

    return weights_best


def test(model, test_set):
    model.eval()
    model.is_eval = True
    print("TESTING NOW ....")
    loss_test, ppl_test, dia_acc_test, ctx_acc_test, trg_acc_test, results, dist1, dist2, avg_len = evaluate(
        model, test_set, ty="test", max_dec_step=50
    )
    print("TEST: {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t\n".format(loss_test, ppl_test, dia_acc_test,
                                                                    ctx_acc_test, trg_acc_test))
    file_summary = config.save_path + "/results.txt"
    with open(file_summary, "w") as f:
        f.write("Loss\tAccuracy\n")
        f.write(
            "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t\n".format(
                loss_test, ppl_test, dia_acc_test, ctx_acc_test, trg_acc_test, dist1, dist2, avg_len
            )
        )
        for r in results:
            f.write(r)


def main():
    set_seed()  # for reproducibility
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    train_set, dev_set, test_set, vocab, dec_num = prepare_data_seq(
        batch_size=config.batch_size
    )

    model = make_model(vocab, dec_num)

    if config.test:
        test(model, test_set)
    else:
        weights_best = train(model, train_set, dev_set)
        model.epoch = 1
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        test(model, test_set)


if __name__ == "__main__":
    main()
