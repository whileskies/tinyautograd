from tinyautograd import *
import gzip
import struct
import numpy as np
from tqdm import tqdm


def read_mnist_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        # 读取图像数据
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows * num_cols)  # 重塑成 (num_images, 28*28)
        images = images / 255.0  # 归一化像素值 [0, 1]
    return images


def read_mnist_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        # 读取文件头（前8个字节）
        magic, num_labels = struct.unpack(">II", f.read(8))
        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def one_hot(labels, num_classes=10):
    # 创建一个全为 0 的矩阵，大小为 (len(labels), num_classes)
    one_hot_encoded = np.zeros((len(labels), num_classes), dtype=np.float32)
    
    # 将对应标签位置设为 1
    one_hot_encoded[np.arange(len(labels)), labels] = 1
    
    return one_hot_encoded


def train_test_split(data, labels, test_size=0.2):
    # 获取数据集的大小
    num_samples = len(data)
    
    # 生成随机索引
    indices = np.random.permutation(num_samples)
    
    # 计算划分的索引
    test_size = int(num_samples * test_size)
    
    # 划分训练集和测试集
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # 根据索引分割数据和标签
    train_data, test_data = data[train_indices], data[test_indices]
    train_labels, test_labels = labels[train_indices], labels[test_indices]
    
    return train_data, test_data, train_labels, test_labels


def train(model, train_data, train_labels, optimizer, epochs=10, batch_size=64):
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        indices = np.random.permutation(len(train_data))
        train_data, train_labels = train_data[indices], train_labels[indices]
        
        for i in tqdm(range(0, len(train_data), batch_size)):
            x_batch = Tensor(train_data[i:i+batch_size])
            y_batch = Tensor(train_labels[i:i+batch_size])

            logists = model(x_batch)
            loss = softmax_cross_entropy(logists, y_batch).sum()
            epoch_loss += loss.data

            predicted = np.argmax(logists.data, axis=1)
            labels = np.argmax(y_batch.data, axis=1)
            correct += np.sum(predicted == labels)
            total += y_batch.data.shape[0]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_accuracy = correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_data)}, Accuracy: {epoch_accuracy*100:.2f}%')


def test(model, test_data, test_labels):
    correct = 0
    total = len(test_data)
    print(total)

    batch_size=64
    for i in tqdm(range(0, total, batch_size)):
        x_batch = Tensor(test_data[i:i+batch_size])
        y_batch = Tensor(test_labels[i:i+batch_size])

        logists = model(x_batch)

        predicted = np.argmax(logists.data, axis=1)
        labels = np.argmax(y_batch.data, axis=1)
        correct += np.sum(predicted == labels)

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


def main():
    image_file_path = './samples/mnist/train-images-idx3-ubyte.gz'
    label_file_path = './samples/mnist/train-labels-idx1-ubyte.gz'
    train_images = read_mnist_images(image_file_path)
    label_images = read_mnist_labels(label_file_path)
    label_images = one_hot(label_images)
    print(train_images.shape)
    train_data, test_data, train_labels, test_labels = train_test_split(train_images, label_images, test_size=0.2)


    model = MLP(28*28, [512], 10, activation_fun=relu)
    opt = SGD(model.parameters())
    train(model, train_data, train_labels, opt, epochs=20)
    test(model, test_data, test_labels)

main()