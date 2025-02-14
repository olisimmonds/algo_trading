import matplotlib.pyplot as plt

def calc_difference(past, future):
    dif = (future-past)/past
    return dif

def breached_threshold(threshold, dict_to_add_to, predicted_difference, i, last_train_price, last_predicted_price, last_actual_price, verbose=0):
    if predicted_difference>threshold:
        if verbose == 1:
            print(i, last_train_price, last_predicted_price, last_actual_price)
        try:
            dict_to_add_to[i].append(calc_difference(last_train_price, last_actual_price))
        except Exception:
            dict_to_add_to[i] = [calc_difference(last_train_price, last_actual_price)]

def plot_skatter(df):
    plt.figure(figsize=(8, 5))
    plt.scatter(df['Predicted Difference'], df['Actual Difference'], marker='o', linestyle='-', color='blue')
    plt.title('Pred vs Act Plot')
    plt.xlabel('Pred Axis')
    plt.ylabel('Act Axis')
    plt.grid(True)
    plt.show()