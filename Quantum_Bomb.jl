using LinearAlgebra
using Random
using Distributions
using Plots
using Printf
using Quante

gr()

# Физические константы
ħ = 1.0545718e-34  # Приведённая постоянная Планка, Дж·с
m_e = 9.10938356e-31  # Масса электрона, кг
e = 1.60217662e-19  # Заряд электрона, Кл
ε0 = 8.854187817e-12  # Электрическая постоянная, Ф/м

# Функция для создания случайного состояния суперпозиции
function create_superposition_state(dim::Int)
    state = rand(ComplexF64, dim)
    return state / norm(state)
end

# Функция для создания плотностной матрицы
function density_matrix(state::Vector{ComplexF64})
    return state * state'
end

# Функция для нормализации вероятностей
function normalize_probabilities(probabilities::Vector{Float64})
    total = sum(probabilities)
    return probabilities / total
end

# Функция для моделирования измерения состояния
function measure_state(rho::Matrix{ComplexF64})
    probabilities = real(diag(rho))
    probabilities = normalize_probabilities(probabilities)
    
    # Проверка на отрицательные значения
    if any(probabilities .< 0)
        error("Некоторые значения вероятностей отрицательны.")
    end

    # Проверка на суммы, не равные 1
    if abs(sum(probabilities) - 1) > 1e-10
        error("Сумма вероятностей не равна 1.")
    end

    dist = Categorical(probabilities)
    return rand(dist)
end

# Функция для создания гамильтониана водорода
function hydrogen_hamiltonian(dim::Int)
    H = zeros(ComplexF64, dim, dim)
    for i in 1:dim
        H[i, i] = -13.6 / i^2  # Энергетические уровни водорода в эВ (приближение)
    end
    return H
end

# Функция для создания квантовой связи между двумя системами
function create_entangled_state(dim::Int)
    state1 = create_superposition_state(dim)
    state2 = create_superposition_state(dim)
    entangled_state = kron(state1, state2)  # Используем произведение Кронекера для создания запутанного состояния
    return entangled_state / norm(entangled_state)
end

# Функция для визуализации состояния
function plot_state(probabilities::Vector{Float64}, title::String)
    bar(probabilities, title=title, xlabel="Индекс", ylabel="Вероятность", legend=false)
end

# Функция для изменения энергии с течением времени
function time_dependent_energy(initial_energy::Float64, final_energy::Float64, current_step::Int, total_steps::Int)
    # Энергия увеличивается экспоненциально
    return initial_energy * ((final_energy / initial_energy) ^ (current_step / total_steps))
end

# Функция для создания операторов Линдблада для N-level системы
function create_lindblad_operators(dim::Int)
    c_ops = Matrix{ComplexF64}[]
    for i in 1:(dim-1)
        for j in (i+1):dim
            op_matrix = zeros(ComplexF64, dim, dim)
            op_matrix[i, j] = 1.0
            push!(c_ops, op_matrix)
        end
    end
    return c_ops
end

# Функция для эволюции плотностной матрицы
function evolve_density_operator(rho::Matrix{ComplexF64}, H::Matrix{ComplexF64}, dt::Float64, c_ops::Vector{Matrix{ComplexF64}}, gamma::Float64)
    H_exp = exp(-im * H * dt)
    rho_evolved = H_exp * rho * H_exp'
    
    # Декогеренция
    for c_op in c_ops
        c_op_term = c_op * rho_evolved * c_op' - 0.5 * (c_op' * c_op * rho_evolved + rho_evolved * c_op' * c_op)
        rho_evolved += gamma * c_op_term
    end

    rho_evolved = (rho_evolved + rho_evolved') / 2
    return rho_evolved
end

# Модель квантовой бомбы
function quantum_bomb_explosion(dim::Int, gamma::Float64, dt::Float64, steps::Int, num_hamiltonians::Int, initial_energy::Float64, final_energy::Float64)
    println("Инициализация квантовой бомбы...")
    
    # Создание запутанного состояния
    entangled_state = create_entangled_state(dim)
    println("Запутанное состояние создано: $entangled_state")

    # Создание плотностной матрицы
    rho = density_matrix(entangled_state)
    println("Плотностная матрица: $rho")
    
    # Создание гамильтонианов водорода для временной эволюции
    Hs = [hydrogen_hamiltonian(dim^2) for _ in 1:num_hamiltonians]

    # Создание операторов Линдблада для декогеренции
    c_ops = create_lindblad_operators(dim^2)

    # Визуализация состояния процессов квантовой бомбы
    anim = @animate for i in 1:steps
        t = i * dt
        if i == 1
            plot_state(abs2.(entangled_state), "Начальное состояние")
        else
            H = Hs[Int(mod(i-1, num_hamiltonians)) + 1]
            rho = evolve_density_operator(rho, H, dt, c_ops, gamma)
            probabilities = real(diag(rho))
            probabilities = normalize_probabilities(probabilities)
            plot_state(probabilities, "Эволюция состояния в шаге $i")
        end
    end

    # Измерение состояния квантовой бомбы
    result = measure_state(rho)
    
    # Энергия на текущем шаге
    current_energy = time_dependent_energy(initial_energy, final_energy, steps, steps)
    println("Процесс измерения завершен. Результат: $result")

    # Сохранение анимации как GIF
    gif(anim, "quantum_bomb_explosion_advanced.gif", fps=3)

    return current_energy
end

# Параметры квантовой бомбы
dimension = 2    # Размерность состояния суперпозиции
gamma = 0.1      # Коэффициент декогеренции
dt = 0.01        # Шаг времени
steps = 100      # Количество шагов эволюции
num_hamiltonians = 10 # Количество гамильтонианов
initial_energy = 4.0 # Начальная энергия в Джоулях
final_energy = 10^48.0 # Конечная энергия в Джоулях

# Вызов функции квантовой бомбы
energy = quantum_bomb_explosion(dimension, gamma, dt, steps, num_hamiltonians, initial_energy, final_energy)
println("Моделирование завершено. Итоговая энергия взрыва квантовой бомбы: $(@sprintf("%.3e", energy)) Джоулей")