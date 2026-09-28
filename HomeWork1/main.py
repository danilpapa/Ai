import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


class AI:
    def __init__(self, input_size=28 * 28, hidden_size=128, output_size=10, lr=0.1):
        # np.random.randn даёт случайные числа около нуля. Умножаем на sqrt(2/784) ≈ 0.05,
        # чтобы они были небольшими. Это проверенный рецепт, чтобы сеть нормально училась.
        self.weight_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)

        # насколько нейрон склонен подавать сигнал".
        # Начинаем с нулей: пусть сначала все будут нейтральными.
        self.bias_hidden = np.zeros(hidden_size)

        # доверие output'a нейронам.
        self.weight_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)

        # 10 чисел
        self.bias_output = np.zeros(output_size)

        self.lr = lr

    def relu(self, z):
        # Правило для проверки: "если совпадения нет, сигнал НЕ подаю".
        # Каждое отрицательное число заменяем на 0, положительные не трогаем.
        return np.maximum(0, z)

    def relu_derivative(self, z):
        # был ли этот нейрон вообще в деле?"
        # 1, если он пдавал сигнал, иначе 0
        return (z > 0).astype(float)

    def softmax(self, z):
        # переводит баллы в ПРОЦЕНТЫ
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot_encoding(self, y_true, num_classes=10):
        # Правильный ответ записан одним числом
        y_true = np.atleast_1d(y_true)
        one_hot = np.zeros((len(y_true), num_classes))

        one_hot[np.arange(len(y_true)), y_true] = 1
        return one_hot

    def cross_entropy(self, probs, y_true):
        # ОЦЕНКА ОШИБКИ
        # Смотрим, сколько процентов output дал шанс ПРАВИЛЬНОЙ цифре:
        # Это число нужно только для того, чтобы следить за прогрессом.
        correct_probs = probs[np.arange(len(y_true)), y_true]
        return -np.mean(np.log(correct_probs + 1e-12))

    def forward_propagation(self, x):
        if x.ndim == 2 and x.shape == (28, 28):
            x = x.reshape(1, -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        self.x = x

        # Для ОДНОГО нейрона и ОДНОЙ картинки считаем:
        #     (пиксель1 × важность1) + (пиксель2 × важность2) + ... + (пиксель784 × важность784)
        # Большая сумма значит "картинка похожа на то, что я ищу".

        # Потом прибавляем bias ("типа характер" каждого нейрона).
        self.z_hidden = x @ self.weight_input_hidden + self.bias_hidden
        # у кого получился минус, тот молчит (0).
        self.hidden = self.relu(self.z_hidden)

        # балл output'a = (нейрон1 × доверие1) + ... + (нейрон128 × доверие128)
        self.z_output = self.hidden @ self.weight_hidden_output + self.bias_output

        # баллы превращаем в проценты.
        output = self.softmax(self.z_output)
        return output

    def predict(self, x):
        output = self.forward_propagation(x)
        return np.argmax(output, axis=1)

    def back_propagation(self, output, y_true):
        # РАЗБОР ПОЛЁТОВ
        # Идём от ответа назад к картинке, поэтому это "обратный проход".
        #
        #   Шаг 1. Смотрим на output: кто как ошибся?
        #   Шаг 2. Меняем доверие output'a к нейронам.
        #   Шаг 3. Считаем вину каждого нейрона.#
        #   Шаг 4. Меняем то, на что смотрят нейрона.
        #   Шаг 5. Все поправки делаем ЧУТЬ-ЧУТЬ (умножаем на lr), чтобы
        #       не сломать то, чему output научился на прошлых картинках.
        #
        # "градиент" — это и есть "вина + в какую сторону поправлять".
        batch_size = output.shape[0]

        # "Промах" каждого output'a = что сказали − что надо было.
        # Плюс — "дал слишком много, надо меньше", минус — "дал слишком мало, надо больше".
        # Делим на размер пачки, чтобы поправка была СРЕДНЕЙ по картинкам, а не суммой.
        d_output = (output - self.one_hot_encoding(y_true)) / batch_size

        # Насколько менять доверие output'a к neuron:
        #     (насколько сильный сигнал нейрона) × (промах output'a)
        grad_w_hidden_output = self.hidden.T @ d_output
        grad_b_output = np.sum(d_output, axis=0)

        # Шаг 3. Вина neuron'a
        d_hidden = d_output @ self.weight_hidden_output.T
        d_hidden *= self.relu_derivative(self.z_hidden)

        # Насколько менять важность пикселя для neuron'a:
        grad_w_input_hidden = self.x.T @ d_hidden
        grad_b_hidden = np.sum(d_hidden, axis=0)

        self.weight_hidden_output -= self.lr * grad_w_hidden_output
        self.bias_output -= self.lr * grad_b_output
        self.weight_input_hidden -= self.lr * grad_w_input_hidden
        self.bias_hidden -= self.lr * grad_b_hidden

    def train(self, X, Y, epochs=5, batch_size=64, X_val=None, Y_val=None):
        history = {"loss": [], "val_acc": []}
        n = X.shape[0]

        for epoch in range(epochs):
            # Перемешиваем картинки и ответы ОДИНАКОВО, чтобы ответы не перепутались.
            perm = np.random.permutation(n)
            X, Y = X[perm], Y[perm]

            epoch_loss = 0.0

            for start in range(0, n, batch_size):
                xb = X[start:start + batch_size]
                yb = Y[start:start + batch_size]

                output = self.forward_propagation(xb)
                # Записываем ошибку для статистики
                epoch_loss += self.cross_entropy(output, yb) * len(yb)
                self.back_propagation(output, yb)

            epoch_loss /= n
            history["loss"].append(epoch_loss)

            msg = f"Эпоха {epoch + 1}/{epochs}: loss = {epoch_loss:.4f}"
            if X_val is not None:
                acc = self.accuracy(X_val, Y_val)
                history["val_acc"].append(acc)
                msg += f", точность на тесте = {acc:.4f}"
            print(msg)

        return history

    def accuracy(self, X, Y):
        return np.mean(self.predict(X) == Y)


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    ai = AI(lr=0.1)

    history = ai.train(X_train, Y_train, epochs=5, batch_size=64, X_val=X_test, Y_val=Y_test)

    print(f"Итоговая точность на тесте: {ai.accuracy(X_test, Y_test):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, len(history["loss"]) + 1), history["loss"], marker="o")
    axes[0].set_title("Loss (кросс-энтропия)")
    axes[0].set_xlabel("Эпоха")
    axes[1].plot(range(1, len(history["val_acc"]) + 1), history["val_acc"], marker="o")
    axes[1].set_title("Точность на тесте")
    axes[1].set_xlabel("Эпоха")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 8, figsize=(14, 2.5))
    for i, ax in enumerate(axes):
        pred = ai.predict(X_test[i])[0]
        ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"сеть: {pred}\nверно: {Y_test[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
