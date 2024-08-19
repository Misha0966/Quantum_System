# Создание накопителя
println("Инициализация накопителя...")

# Константы для накопителя
max_capacity = 1.181e+22  # Максимальная ёмкость накопителя в Джоулях
current_energy = 0.0  # Начальная энергия в накопителе

# Функция для зарядки накопителя
function charge_accumulator!(energy::Float64)
    global current_energy
    if current_energy + energy <= max_capacity
        current_energy += energy
        println("Накопитель заряжен. Текущая энергия: $current_energy Дж")
    else
        println("Превышена ёмкость накопителя. Зарядка не возможна.")
    end
end

# Функция для разрядки накопителя
function discharge_accumulator!(required_energy::Float64)
    global current_energy
    if current_energy >= required_energy
        current_energy -= required_energy
        println("Разрядка выполнена. Остаток энергии: $current_energy Дж")
        return required_energy
    else
        println("Недостаточно энергии для выполнения разрядки.")
        return 0.0
    end
end

# Пример работы накопителя
energy = 1.181e+21  # Энергия от квантового реактора
energy_to_store = energy  # Заряжаем накопитель

charge_accumulator!(energy_to_store)  # Заряжаем накопитель

# Пример: потребность в энергии для распределительной сети
energy_needed = 1e21
discharge_accumulator!(energy_needed)  # Выполняем разрядку