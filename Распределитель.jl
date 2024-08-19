# Создание распределителя
println("Инициализация распределителя...")

# Функция для распределения энергии по потребностям
function distribute_energy!(total_energy::Float64, demands::Vector{Float64})
    current_energy = total_energy
    distributed_energy = []
    for demand in demands
        if current_energy >= demand
            push!(distributed_energy, demand)
            current_energy -= demand
        else
            push!(distributed_energy, current_energy)
            current_energy = 0.0
            break
        end
    end
    println("Энергия распределена по запросам: $distributed_energy")
    println("Остаток энергии в накопителе: $current_energy Дж")
    return distributed_energy
end

# Пример работы распределителя
initial_energy = 1e9  # Пример начального значения энергии
energy_demands = [2e8, 3e8, 5e7, 11e7, 2e8, 5e7, 7e7, 2e7]  # Запросы на энергию от разных потребителей

# Распределение энергии
distributed = distribute_energy!(initial_energy, energy_demands)