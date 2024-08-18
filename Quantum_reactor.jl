using LinearAlgebra  # Для работы с линейной алгеброй
using Random  # Для генерации случайных чисел
using Distributions  # Для работы с распределениями
using Plots  # Для построения графиков
using Printf  # Для форматированного вывода
using Quante  # Для квантовых вычислений

gr()  # Выбор бэкенда для построения графиков

# Физические константы
ħ = 1.0545718e-34  # Приведённая постоянная Планка, Дж·с
m_e = 9.10938356e-31  # Масса электрона, кг
e = 1.60217662e-19  # Заряд электрона, Кл
ε0 = 8.854187817e-12  # Электрическая постоянная, Ф/м

# Функция для создания случайного состояния суперпозиции
function create_superposition_state(dim::Int)
    state = rand(ComplexF64, dim)  # Генерация случайного комплексного вектора
    return state / norm(state)  # Нормализация вектора
end

# Функция для создания плотностной матрицы
function density_matrix(state::Vector{ComplexF64})
    return state * state'  # Создание плотностной матрицы из вектора состояния
end

# Функция для нормализации вероятностей
function normalize_probabilities(probabilities::Vector{Float64})
    total = sum(probabilities)  # Суммирование всех вероятностей
    return probabilities / total  # Нормализация вероятностей
end

# Функция для моделирования измерения состояния
function measure_state(rho::Matrix{ComplexF64})
    probabilities = real(diag(rho))  # Извлечение вероятностей из диагональных элементов плотностной матрицы
    probabilities = normalize_probabilities(probabilities)  # Нормализация вероятностей
    
    # Проверка на отрицательные значения
    if any(probabilities .< 0)
        error("Некоторые значения вероятностей отрицательны.")
    end

    # Проверка на суммы, не равные 1
    if abs(sum(probabilities) - 1) > 1e-10
        error("Сумма вероятностей не равна 1.")
    end

    dist = Categorical(probabilities)  # Создание категориального распределения
    return rand(dist)  # Случайный выбор согласно распределению
end

# Функция для создания гамильтониана водорода
function hydrogen_hamiltonian(dim::Int)
    H = zeros(ComplexF64, dim, dim)  # Инициализация матрицы Гамильтониана
    for i in 1:dim
        H[i, i] = -13.6 / i^2  # Энергетические уровни водорода в эВ (приближение)
    end
    return H
end

# Функция для создания квантовой связи между двумя системами
function create_entangled_state(dim::Int)
    state1 = create_superposition_state(dim)  # Создание первого состояния суперпозиции
    state2 = create_superposition_state(dim)  # Создание второго состояния суперпозиции
    entangled_state = kron(state1, state2)  # Используем произведение Кронекера для создания запутанного состояния
    return entangled_state / norm(entangled_state)  # Нормализация запутанного состояния
end

# Функция для визуализации состояния
function plot_state(probabilities::Vector{Float64}, title::String)
    bar(probabilities, title=title, xlabel="Индекс", ylabel="Вероятность", legend=false)  # Построение гистограммы вероятностей
end

# Функция для изменения энергии с течением времени
function time_dependent_energy(start_energy::Float64, final_energy::Float64, current_step::Int, total_steps::Int)
    # Энергия увеличивается экспоненциально
    return start_energy * ((final_energy / start_energy) ^ (current_step / total_steps))
end

# Функция для создания операторов Линдблада для N-level системы
function create_lindblad_operators(dim::Int)
    c_ops = Matrix{ComplexF64}[]  # Инициализация списка операторов Линдблада
    for i in 1:(dim-1)
        for j in (i+1):dim
            op_matrix = zeros(ComplexF64, dim, dim)  # Создание нулевой матрицы
            op_matrix[i, j] = 1.0  # Заполнение матрицы значением
            push!(c_ops, op_matrix)  # Добавление оператора в список
        end
    end
    return c_ops
end

# Функция для эволюции плотностной матрицы
function evolve_density_operator(rho::Matrix{ComplexF64}, H::Matrix{ComplexF64}, dt::Float64, c_ops::Vector{Matrix{ComplexF64}}, gamma::Float64)
    H_exp = exp(-im * H * dt)  # Вычисление экспоненты Гамильтониана
    rho_evolved = H_exp * rho * H_exp'  # Эволюция плотностной матрицы
    
    # Декогеренция
    for c_op in c_ops
        c_op_term = c_op * rho_evolved * c_op' - 0.5 * (c_op' * c_op * rho_evolved + rho_evolved * c_op' * c_op)
        rho_evolved += gamma * c_op_term  # Применение оператора Линдблада
    end

    rho_evolved = (rho_evolved + rho_evolved') / 2  # Обеспечение эрмитовости плотностной матрицы
    return rho_evolved
end

# Модель квантового реактора
function quantum_reactor_explosion(dim::Int, gamma::Float64, dt::Float64, steps::Int, num_hamiltonians::Int, start_energy::Float64, final_energy::Float64)
    println("Инициализация квантового реактора...")
    
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

    # Визуализация состояния процессов квантового реактора
    anim = @animate for i in 1:steps
        t = i * dt
        if i == 1
            plot_state(abs2.(entangled_state), "Начальное состояние")  # Построение начального состояния
        else
            H = Hs[Int(mod(i-1, num_hamiltonians)) + 1]  # Выбор текущего Гамильтониана
            rho = evolve_density_operator(rho, H, dt, c_ops, gamma)  # Эволюция плотностной матрицы
            probabilities = real(diag(rho))  # Извлечение вероятностей
            probabilities = normalize_probabilities(probabilities)  # Нормализация вероятностей
            plot_state(probabilities, "Эволюция состояния в шаге $i")  # Построение текущего состояния
        end
    end

    # Измерение состояния квантового реактора
    result = measure_state(rho)
    
    # Энергия на текущем шаге
    current_energy = time_dependent_energy(start_energy, final_energy, steps, steps)
    println("Процесс измерения завершен. Результат: $result")

    # Сохранение анимации как GIF
    gif(anim, "quantum_reactor_explosion_advanced.gif", fps=1)

    return current_energy
end

# Параметры квантового реактора
dimension = 4    # Размерность состояния суперпозиции
gamma = 0.5      # Коэффициент декогеренции
dt = 0.1     # Шаг времени
steps = 45      # Количество шагов эволюции
num_hamiltonians = 5 # Количество гамильтонианов
start_energy = 1.0 # Начальная энергия в Джоулях
final_energy = 2^70.0 # Конечная энергия в Джоулях

# Вызов функции квантового реактора
energy = quantum_reactor_explosion(dimension, gamma, dt, steps, num_hamiltonians, start_energy, final_energy)
println("Моделирование завершено. Итоговая энергия выработаная квантовым реактором: $(@sprintf("%.3e", energy)) Джоулей")
