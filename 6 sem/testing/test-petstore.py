import pytest
import requests

BASE_URL = "https://petstore.swagger.io/v2"


# --- Тесты для работы с Pet ---
class TestPet:
    def test_add_pet(self) -> int:
        """Тестирование добавления нового питомца (POST /pet)"""
        pet_data = {
            "id": 123,
            "name": "Rex",
            "status": "available"
        }
        response = requests.post(f"{BASE_URL}/pet", json=pet_data)

        assert response.status_code == 200
        assert response.json()["id"] == pet_data["id"]
        assert response.json()["name"] == pet_data["name"]
        assert response.json()["status"] == pet_data["status"]

        #return pet_data["id"]  # Возвращаем ID для использования в других тестах

    def test_get_pet(self, pet_id):
        """Тестирование получения информации о питомце (GET /pet/{petId})"""
        response = requests.get(f"{BASE_URL}/pet/{pet_id}")

        assert response.status_code == 200
        assert response.json()["id"] == pet_id
        assert "name" in response.json()
        assert "status" in response.json()

    def test_update_pet(self, pet_id):
        """Тестирование обновления питомца (PUT /pet)"""
        updated_data = {
            "id": pet_id,
            "name": "Rex Updated",
            "status": "sold"
        }
        response = requests.put(f"{BASE_URL}/pet", json=updated_data)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == pet_id
        assert data["photoUrls"] == []
        assert data["tags"] == []

    def test_delete_pet(self, pet_id):
        """Тестирование удаления питомца (DELETE /pet/{petId})"""
        response = requests.delete(f"{BASE_URL}/pet/{pet_id}")
        assert response.status_code == 200

        # Проверяем, что питомец удален
        get_response = requests.get(f"{BASE_URL}/pet/{pet_id}")
        assert get_response.status_code == 404

    def test_upload_image_to_pet(self, pet_id):
        """POST /pet/{petId}/uploadImage — Загрузка изображения питомцу"""
        files = {'file': open("cat.png" , "rb")}
        response = requests.post(f"{BASE_URL}/pet/{pet_id}/uploadImage", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["type"] == "unknown"
        assert "File uploaded to" in data["message"]

    def test_find_pets_by_status(self):
        """GET /pet/findByStatus — Поиск питомцев по статусу"""
        statuses = ["available", "pending", "sold"]
        for status in statuses:
            response = requests.get(f"{BASE_URL}/pet/findByStatus?status={status}")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            if data:
                for pet in data:
                    assert pet["status"] == status


# --- Тесты для работы с Store ---
class TestStore:
    def test_create_order(self, pet_id):
        """Тестирование создания заказа (POST /store/order)"""
        order_data = {
            "id": 1,
            "petId": pet_id,
            "quantity": 1,
            "status": "placed"
        }
        response = requests.post(f"{BASE_URL}/store/order", json=order_data)

        assert response.status_code == 200
        assert response.json()["petId"] == pet_id
        assert response.json()["status"] == "placed"

        #return response.json()["id"]  # Возвращаем ID заказа

    def test_get_order(self, order_id):
        """Тестирование получения заказа (GET /store/order/{orderId})"""
        response = requests.get(f"{BASE_URL}/store/order/{order_id}")

        assert response.status_code == 200
        assert response.json()["id"] == order_id

    def test_delete_order(self, order_id):
        """Тестирование удаления заказа (DELETE /store/order/{orderId})"""
        response = requests.delete(f"{BASE_URL}/store/order/{order_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["type"] == "unknown"
        assert data["message"] == str(order_id)

    def test_get_store_inventory(self):
        """GET /store/inventory — Получение инвентаря магазина"""
        response = requests.get(f"{BASE_URL}/store/inventory")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        for key in data:
            assert isinstance(key, str)
            assert isinstance(data[key], int)


# --- Тесты для работы с User ---
class TestUser:
    def test_create_user(self):
        """Тестирование создания пользователя (POST /user)"""
        user_data = {
            "id": 1,
            "username": "test_user",
            "email": "user@test.com",
            "password": "test123"
        }
        response = requests.post(f"{BASE_URL}/user", json=user_data)

        assert response.status_code == 200
        assert response.json()["code"] == 200
        assert response.json()["type"] == "unknown"
        assert response.json()["message"] == "1"

        #return user_data["username"]

    def test_get_user(self, username):
        """Тестирование получения пользователя (GET /user/{username})"""
        response = requests.get(f"{BASE_URL}/user/{username}")

        assert response.status_code == 200
        assert response.json()["username"] == username

    def test_update_user(self, username):
        """PUT /user/{username} — Обновление пользователя"""
        updated_data = {
            "username": username,
            "firstName": "UpdatedFirstName",
            "lastName": "UpdatedLastName",
            "email": "updated@example.com"
        }
        response = requests.put(f"{BASE_URL}/user/{username}", json=updated_data)

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["type"] == "unknown"
        assert data["message"].isdigit()  # ID обновленного пользователя

    def test_delete_user(self, username):
        """DELETE /user/{username} — Удаление пользователя"""
        response = requests.delete(f"{BASE_URL}/user/{username}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["type"] == "unknown"
        assert data["message"] == username

        # Проверяем, что пользователь удален
        get_response = requests.get(f"{BASE_URL}/user/{username}")
        assert get_response.status_code == 404

    def test_login_user(self, username):
        """GET /user/login — Логин пользователя"""
        params = {"username": username, "password": "test123"}
        response = requests.get(f"{BASE_URL}/user/login", params=params)

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["type"] == "unknown"
        assert "logged in user session:" in data["message"]

    def test_logout_user(self):
        """GET /user/logout — Выход пользователя"""
        response = requests.get(f"{BASE_URL}/user/logout")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["type"] == "unknown"
        assert data["message"] == "ok"

    def test_create_users_with_list(self):
        """POST /user/createWithList — Создание нескольких пользователей списком"""
        users = [
            {"id": 1001, "username": "user_list_1", "email": "list1@example.com"},
            {"id": 1002, "username": "user_list_2", "email": "list2@example.com"}
        ]
        response = requests.post(f"{BASE_URL}/user/createWithList", json=users)

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["type"] == "unknown"
        assert data["message"] == "ok"

    def test_create_users_with_array(self):
        """POST /user/createWithArray — Создание пользователей массивом"""
        users = [
            {"id": 2001, "username": "user_array_1", "email": "array1@example.com"},
            {"id": 2002, "username": "user_array_2", "email": "array2@example.com"}
        ]
        response = requests.post(f"{BASE_URL}/user/createWithArray", json=users)

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["type"] == "unknown"
        assert data["message"] == "ok"


# --- Тесты обработки ошибок ---
class TestErrorHandling:
    def test_get_nonexistent_pet(self):
        """Тестирование запроса несуществующего питомца"""
        response = requests.get(f"{BASE_URL}/pet/999999")
        assert response.status_code == 404
        data = response.json()
        assert data["code"] == 1
        assert data["type"] == "error"
        assert data["message"] == "Pet not found"

    def test_create_invalid_pet(self):
        """Тестирование создания питомца с невалидными данными"""
        response = requests.post(f"{BASE_URL}/pet", json={"invalid": "data"})
        assert response.status_code == 500

    def test_delete_nonexistent_order(self):
        """Тестирование удаления несуществующего заказа"""
        response = requests.delete(f"{BASE_URL}/store/order/999999")
        assert response.status_code == 404
        data = response.json()
        assert data["code"] == 404
        assert data["type"] == "unknown"
        assert data["message"] == "Order Not Found"


# --- Фикстуры ---
@pytest.fixture
def pet_id():
    pet_data = {
        "id": 123,
        "name": "Rex",
        "status": "available"
    }
    requests.post(f"{BASE_URL}/pet", json=pet_data)
    yield pet_data["id"]
    requests.delete(f"{BASE_URL}/pet/{pet_data['id']}")

@pytest.fixture
def order_id(pet_id):
    order_data = {
        "id": 1,
        "petId": pet_id,
        "quantity": 1,
        "status": "placed"
    }
    response = requests.post(f"{BASE_URL}/store/order", json=order_data)
    yield response.json()["id"]
    requests.delete(f"{BASE_URL}/store/order/{response.json()['id']}")

@pytest.fixture
def username():
    user_data = {
        "id": 1,
        "username": "test_user",
        "email": "user@test.com",
        "password": "test123"
    }
    requests.post(f"{BASE_URL}/user", json=user_data)
    yield user_data["username"]
    requests.delete(f"{BASE_URL}/user/{user_data['username']}")