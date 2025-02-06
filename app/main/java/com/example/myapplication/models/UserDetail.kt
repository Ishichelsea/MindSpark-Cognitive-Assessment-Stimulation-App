package com.example.myapplication.models

import java.io.Serializable

data class UserDetail(
//    val Game_2: Map<String, GameSession>, // The key is the game session identifier
    val age: String = "",
    val id: String= "",
    val password: String = "",
    val sessions: Map<String,Any> = emptyMap() // The key is the session identifier
):Serializable
